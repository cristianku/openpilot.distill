package ch.bodhi.distill

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ServerAddressTest {
    @Test fun validatesPlainURL() {
        assertEquals("https://example.com:8766", ServerAddress.parse("https://example.com:8766/", false))
        assertEquals("http://192.168.1.10:8766", ServerAddress.parse("http://192.168.1.10:8766", true))
        assertNull(ServerAddress.parse("http://192.168.1.10:8766", false))
        assertNull(ServerAddress.parse("https://example.com/?token=secret", false))
        assertNull(ServerAddress.parse("https://user:password@example.com", false))
        assertNull(ServerAddress.parse("javascript:alert(1)", true))
    }
}
