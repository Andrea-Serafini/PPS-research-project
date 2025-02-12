val scala3Version = "3.3.1"

lazy val root = project
  .in(file("."))
  .settings(
    name := "onnx-scala",
    version := "1.0.0-SNAPSHOT",

    scalaVersion := scala3Version,

    libraryDependencies += "com.github.sbt" % "junit-interface" % "0.13.2" % Test,
    libraryDependencies += "org.emergent-order" %% "onnx-scala-backends" % "0.17.0"
  )