plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Release signing comes from the environment (GitHub secrets in CI, see README.md); without it the
// release APK is signed with the debug key of the machine that builds it.
val releaseKeystore: String? = System.getenv("DISTILL_KEYSTORE")?.takeIf { it.isNotBlank() }

android {
    namespace = "ch.bodhi.distill"
    compileSdk = 35

    defaultConfig {
        applicationId = "ch.bodhi.distill"
        minSdk = 26
        targetSdk = 35
        // Set by CI from the run number: -PdistillVersionCode=… -PdistillVersionName=…
        versionCode = (findProperty("distillVersionCode") as String?)?.toInt() ?: 1
        versionName = (findProperty("distillVersionName") as String?) ?: "0.2.0-dev"
    }

    signingConfigs {
        create("release") {
            if (releaseKeystore != null) {
                storeFile = file(releaseKeystore)
                storePassword = System.getenv("DISTILL_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("DISTILL_KEY_ALIAS")
                keyPassword = System.getenv("DISTILL_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName(if (releaseKeystore != null) "release" else "debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
    testImplementation("junit:junit:4.13.2")
    // The real org.json: the one of android.jar is a stub in JVM unit tests.
    testImplementation("org.json:json:20240303")
}
