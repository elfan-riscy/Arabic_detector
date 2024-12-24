# Basic ProGuard rules
-keep public class * {
    public *;
}
-dontwarn **
-keep class org.tensorflow.** { *; }
