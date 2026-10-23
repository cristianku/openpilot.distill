import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.github.triplet.play")
}

// Release signing comes from the environment: helpers/build_android.sh passes the key of the server
// (~/.config/distill-android, never in git). For Google Play it is the upload key.
val releaseKeystore: String? = System.getenv("DISTILL_KEYSTORE")?.takeIf { it.isNotBlank() }

android {
    namespace = "ch.bodhi.distill"
    compileSdk = 36

    defaultConfig {
        applicationId = "ch.bodhi.distill"
        minSdk = 26
        // Google Play takes new apps and updates only with a recent target.
        targetSdk = 36
        // Set by helpers/build_android.sh from the commits of master: -PdistillVersionCode=… -PdistillVersionName=…
        versionCode = (findProperty("distillVersionCode") as String?)?.toInt() ?: 1
        versionName = (findProperty("distillVersionName") as String?) ?: "0.3.0-dev"
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
            signingConfig = signingConfigs.getByName("release")
            // R8: a smaller app, and the code obfuscated as Google Play expects; its mapping goes to Play
            // with the bundle, for readable crash reports.
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
    }
    lint {
        // The toolchain is pinned on purpose (build.gradle.kts); these only report newer versions.
        disable += "OldTargetApi"
        disable += "GradleDependency"
        disable += "AndroidGradlePluginVersion"
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

// Google Play: the key of the service account comes from helpers/build_android.sh (--play, --listing), from
// ~/.config/distill-android/play-service-account.json; the track can be changed with --track.
play {
    System.getenv("DISTILL_PLAY_CREDENTIALS")?.takeIf { it.isNotBlank() }?.let { serviceAccountCredentials.set(file(it)) }
    track.set("internal")
    defaultToAppBundles.set(true)
}

// A release is never signed with a debug key or left unsigned: Google Play and the phones that have the app
// would refuse it.
tasks.matching { it.name == "preReleaseBuild" }.configureEach {
    doFirst {
        check(releaseKeystore != null) {
            "Release signing is missing: build with helpers/build_android.sh, which passes the key of the server " +
                "(DISTILL_KEYSTORE, DISTILL_KEYSTORE_PASSWORD, DISTILL_KEY_ALIAS, DISTILL_KEY_PASSWORD)"
        }
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2025.12.01"))
    implementation("androidx.activity:activity-compose:1.11.0")
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
