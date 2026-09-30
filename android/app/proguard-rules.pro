# Native llama.cpp methods are called from JNI — keep their names.
-keepclasseswithmembers class * {
    native <methods>;
}
-keep class com.example.llamachat.inference.** { *; }
