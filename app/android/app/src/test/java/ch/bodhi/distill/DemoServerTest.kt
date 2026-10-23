package ch.bodhi.distill

import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test

/** The demo server answers the API of distill-web in the shapes the app reads (the same parsers as a real
 * server), and its jobs follow the rules of the site: one GPU job, one comma job at a time. */
class DemoServerTest {
    private var now = 1_792_000_000_000L
    private lateinit var session: String

    @Before fun signIn() {
        DemoServer.clock = { now }
        DemoServer.reset()
        session = DemoServer.answer("POST", "/api/auth/login", null,
            JSONObject().put("user", DemoServer.USER).put("password", DemoServer.PASSWORD)).cookies.single().substringBefore(";")
    }

    @After fun realClock() {
        DemoServer.clock = System::currentTimeMillis
        DemoServer.reset()
    }

    private fun get(path: String) = JSONObject(DemoServer.answer("GET", path, session, null).text)
    private fun post(path: String, body: JSONObject = JSONObject()) = JSONObject(DemoServer.answer("POST", path, session, body).text)
    private fun status() = DistillApi.parseStatus(get("/api/status"))

    @Test fun needsTheDemoSignIn() {
        val state = DistillApi.parseInfo(JSONObject(DemoServer.answer("GET", "/api/auth/state", null, null).text))!!
        assertEquals(ServerInfo(configured = true, signedIn = false, api = DistillApi.API_VERSION), state)
        try {
            DemoServer.answer("GET", "/api/status", null, null)
            fail("the status without a session")
        } catch (_: SignInNeeded) {
        }
        try {
            DemoServer.answer("POST", "/api/auth/login", null, JSONObject().put("user", "demo").put("password", "nope"))
            fail("a wrong password")
        } catch (exception: SignInNeeded) {
            assertTrue(exception.message!!.contains("password demo"))
        }
        assertTrue(DistillApi.parseInfo(get("/api/auth/state"))!!.signedIn)
    }

    @Test fun startsFollowsAndFinishesATraining() {
        val first = status()
        assertTrue(first.jobs.isEmpty())
        assertEquals("", first.reason("train"))
        assertEquals("Tesla V100", first.gpus.single().name)
        val runs = get("/api/runs").getJSONArray("runs")
        assertEquals(3, runs.length())
        assertTrue(DistillApi.parseRun(runs.getJSONObject(0)).resumable)

        post("/api/run/train", JSONObject().put("mode", "fresh").put("steps", 1000))
        now += 90_000
        val running = status()
        val job = running.jobs.single()
        assertEquals("Training", job.label)
        assertEquals(50.0, job.percent!!, 0.1)
        assertEquals(500, job.done)
        assertEquals(1000, running.training.single().total)
        // One GPU job at a time, as on the server.
        assertTrue(running.reason("eval").startsWith("the GPU is busy"))
        try {
            post("/api/run/eval", JSONObject().put("model", "stock"))
            fail("a second GPU job")
        } catch (exception: ServerRefused) {
            assertTrue(exception.message!!.contains("GPU is busy"))
        }
        assertEquals("", running.reason("install"))

        now += 120_000
        assertTrue(status().jobs.isEmpty())
        val finished = DistillApi.parseRun(get("/api/runs").getJSONArray("runs").getJSONObject(0))
        assertTrue(finished.hasPt)
        assertEquals(1000, finished.stepsTotal)
    }

    @Test fun stopsAJob() {
        val job = post("/api/run/eval", JSONObject().put("model", "stock")).getJSONObject("job")
        assertEquals(1, status().jobs.size)
        post("/api/jobs/" + job.getString("id") + "/stop")
        assertTrue(status().jobs.isEmpty())
        try {
            post("/api/jobs/" + job.getString("id") + "/stop")
            fail("a job that is gone")
        } catch (_: ServerRefused) {
        }
    }

    @Test fun anInstallSavesTheOriginalModel() {
        assertEquals(false, status().comma.backup)
        post("/api/run/install", JSONObject().put("run", "x").put("confirm", "INSTALL"))
        assertTrue(status().reason("restore").startsWith("the comma is busy"))
        now += 60_000
        assertEquals(true, status().comma.backup)
        assertTrue(post("/api/comma/check").getJSONObject("comma").getBoolean("reachable"))
    }

    @Test fun signsOut() {
        val cookie = DemoServer.answer("POST", "/api/auth/logout", session, JSONObject()).cookies.single()
        assertTrue(cookie.contains("Max-Age=0"))
    }
}

/** GET /api/auth/state of servers of different ages. */
class ServerInfoTest {
    @Test fun readsTheVersionOfTheApi() {
        val old = DistillApi.parseInfo(JSONObject("""{"configured": true, "signed_in": false}"""))!!
        assertEquals(1, old.api)
        assertEquals("", old.problem)
        val newer = DistillApi.parseInfo(JSONObject("""{"configured": true, "signed_in": true, "api": 2}"""))!!
        assertTrue(newer.problem.contains("update Distill"))
        assertFalse(DistillApi.parseInfo(JSONObject("""{"configured": false, "signed_in": false, "api": 1}"""))!!.configured)
        assertNull(DistillApi.parseInfo(JSONObject("""{"ok": true}""")))
    }
}
