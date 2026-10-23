package ch.bodhi.distill

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** status.json and runs.json were produced by web/server.py (App.status_payload, App.runs_payload) on a test
 * instance with a fake comma: if the server changes a field the app reads, this test says so. */
class ApiContractTest {
    private fun fixture(name: String) = JSONObject(javaClass.classLoader!!.getResource(name)!!.readText())

    @Test fun readsTheStatus() {
        val status = DistillApi.parseStatus(fixture("status.json"))
        assertEquals(listOf("GPU", "CPU", "Memory", "Disk"), status.metrics.map { it.title })
        assertEquals("Tesla V100", status.gpus.single().name)
        assertEquals(32, status.gpus.single().memoryGb)
        // Only the running job, with its progress and its stop button.
        val job = status.jobs.single()
        assertEquals("Training", job.label)
        assertEquals(42.0, job.percent!!, 0.0)
        assertEquals(5000, job.total)
        assertTrue(job.stoppable)
        // The reasons of the server are shown under the buttons.
        for (action in listOf("train", "eval", "install", "restore")) assertTrue(action, status.actions.containsKey(action))
        assertTrue(status.reason("eval").startsWith("this version of the code"))
        assertEquals("server status not read yet", status.reason("no_such_action"))
        assertEquals(true, status.comma.reachable)
        assertEquals(true, status.comma.bundleDefault)
        assertEquals(false, status.comma.backup)
        assertFalse(status.comma.onroad)
    }

    @Test fun readsTheRuns() {
        val runs = (fixture("runs.json").getJSONArray("runs")).let { array ->
            (0 until array.length()).map { DistillApi.parseRun(array.getJSONObject(it)) }
        }
        val finished = runs.single { it.hasPt }
        assertFalse(finished.resumable)
        assertEquals(20, finished.stepsDone)
        val stopped = runs.single { !it.hasPt }
        assertTrue(stopped.hasTrainingPt)
        assertTrue(stopped.resumable)
    }
}
