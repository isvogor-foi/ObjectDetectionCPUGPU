To run from Eclipse add OpenCV (jar and native libs) and VM arguments:
-Djava.library.path=.:/hr/foi/tiwo:/home/ivan/Dev/opencv-2.4.10/bin/opencv-2410.jar:/hr/foi/tiwo:/home/ivan/Dev/opencv-2.4.10/lib

For CPU object detection:
 - compile C++ code like: g++ -o libDetectCpu.so -lc -shared -I/usr/lib/jvm/java-7-openjdk-amd64/include/ -I/usr/lib/jvm/java-7-openjdk-amd64/include/linux/ arrow_cpu.cpp -fPIC `pkg-config opencv --cflags --libs`
 - in ObjDec.java uncomment: System.loadLibrary("DetectCpu");

For GPU object detection:
 - combile C++ code like: g++ -o libDetectGpu.so -lc -shared -I/usr/lib/jvm/java-7-openjdk-amd64/include/ -I/usr/lib/jvm/java-7-openjdk-amd64/include/linux/ arrow_gpu.cpp -fPIC `pkg-config opencv --cflags --libs`
 - in ObjDec.java uncomment: System.loadLibrary("DetectGpu");

For GPU object detection:

The project depends on:
 - OpenCV
 - OpenCL
 
For OpenCL CPU Haar detection use haarcascade_10.xml or cascade_boost.xml, but for GPU use haarcascade_10.xml (for some reason it doesn't support boost layout for XML Haar classifier)