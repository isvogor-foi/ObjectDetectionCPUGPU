#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ocl/ocl.hpp"
#include "hr_foi_tiwo_ObjDec.h"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

#define LOOP_NUM 10
#define MAX_THREADS 10

// BUILD ME AS:
// g++ -o libDetectCpu.so -lc -shared -I/usr/lib/jvm/java-7-openjdk-amd64/include/ -I/usr/lib/jvm/java-7-openjdk-amd64/include/linux/ arrow_cpu.cpp -fPIC `pkg-config opencv --cflags --libs`
///////////////////////////single-threading arrows detecting///////////////////////////////

const static Scalar colors[] =  { CV_RGB(0,0,255),
                                  CV_RGB(0,128,255),
                                  CV_RGB(0,255,255),
                                  CV_RGB(0,255,0),
                                  CV_RGB(255,128,0),
                                  CV_RGB(255,255,0),
                                  CV_RGB(255,0,0),
                                  CV_RGB(255,0,255)
                                } ;

int64 work_begin[MAX_THREADS] = {0};
int64 work_total[MAX_THREADS] = {0};
string inputName, outputName, cascadeName;

static void workBegin(int i = 0)
{
    work_begin[i] = getTickCount();
}

static void workEnd(int i = 0)
{
    work_total[i] += (getTickCount() - work_begin[i]);
}

static double getTotalTime(int i = 0)
{
    return work_total[i] /getTickFrequency() * 1000.;
}

static void detectCPU( Mat& img, vector<Rect>& arrows, CascadeClassifier& cascade, double scale, bool calTime);
static void Draw(Mat& img, vector<Rect>& arrows, double scale);

static int arrowdetect_one_thread(bool useCPU, double scale, Mat image)
{
    CvCapture* capture = 0;

    CascadeClassifier  cpu_cascade;
    
    cout<<"Started..."<<endl;

    if( !cpu_cascade.load(cascadeName) )
    {
        cout << "ERROR: Could not load classifier cascade: " << cascadeName << endl;
        return EXIT_FAILURE;
    }
    
    cout<<"Loaded..."<<endl;
    
    cout << "In image read " << image.size() << endl;
    vector<Rect> arrows;
    vector<Rect> ref_rst;
    cout << "loops: ";
    
    for(int i = 0; i <= LOOP_NUM; i++)
    {
	cout << i << ", ";
	detectCPU(image, arrows, cpu_cascade, scale, i!=0);
    }
    cout << "done!" << endl;
    cout << "average CPU time (noCamera) : ";
    cout << getTotalTime() / LOOP_NUM << " ms" << endl;
    cout << "Detected elements: "<<arrows.size()<<endl;

    Draw(image, arrows, scale);
    waitKey(0);
    
    cout<< "single-threaded sample has finished" <<endl;
    return 0;
}

// ---------------------------- NATIVE

JNIEXPORT void JNICALL Java_hr_foi_tiwo_ObjDec_sendImageForProcessing(JNIEnv *env, jclass cls, jlong image, jint width, jint height){  
  
  cout<<"Hello from C++"<<endl;
  //cascadeName = "cascade_boost.xml";
  cascadeName = "haarcascade_10.xml";
  Mat* mat = (Mat*) image;
  arrowdetect_one_thread(true, 1.0, *mat);

}

// ---------------------------- END NATIVE


void detectCPU( Mat& img, vector<Rect>& arrows, CascadeClassifier& cascade, double scale, bool calTime)
{
    if(calTime) workBegin();
      Mat cpu_gray, cpu_smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
      cvtColor(img, cpu_gray, CV_BGR2GRAY);
      resize(cpu_gray, cpu_smallImg, cpu_smallImg.size(), 0, 0, INTER_LINEAR);
      equalizeHist(cpu_smallImg, cpu_smallImg);
      cascade.detectMultiScale(cpu_smallImg, arrows, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30), Size(0, 0));
    if(calTime) workEnd();
}

void Draw(Mat& img, vector<Rect>& arrows, double scale)
{
    int i = 0;
    for( vector<Rect>::const_iterator r = arrows.begin(); r != arrows.end(); r++, i++ )
    {
        Point center;
        Scalar color = colors[i%8];
        int radius;
        center.x = cvRound((r->x + r->width*0.5)*scale);
        center.y = cvRound((r->y + r->height*0.5)*scale);
        radius = cvRound((r->width + r->height)*0.25*scale);
        circle( img, center, radius, color, 3, 8, 0 );
    }
    if( !outputName.empty() ) imwrite( outputName, img );
    if( abs(scale-1.0)>.001 )
    {
        resize(img, img, Size((int)(img.cols/scale), (int)(img.rows/scale)));
    }
    imwrite("res-from-cpp.png", img);
}
