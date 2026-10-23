# R8 rules of the release build. The app uses no reflection: Compose, coroutines and org.json (part of
# Android) need nothing more than the rules their libraries bring.
# Keep line numbers in the crash reports of Google Play (the mapping file goes up with the bundle).
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile
