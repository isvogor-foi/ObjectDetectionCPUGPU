#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ocl/ocl.hpp"
#include "hr_foi_tiwo_ObjDec.h"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

#define LOOP_NUM 0
#define MAX_THREADS 10
#define MILL 1e6

// BUILD ME AS:
// g++ -o libDetectGpu.so -lc -shared -I/usr/lib/jvm/java-7-openjdk-amd64/include/ -I/usr/lib/jvm/java-7-openjdk-amd64/include/linux/ arrow_gpu.cpp -fPIC `pkg-config opencv --cflags --libs`
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

struct timespec before, after;


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

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) 
  {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000 + end.tv_nsec-start.tv_nsec;
  } else 
  {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  cout<<"Execution time: "<<(temp.tv_nsec / MILL)<<"ms"<<endl;
  
  return temp;
}

static void detect( Mat& img, vector<Rect>& arrows, ocl::OclCascadeClassifier& cascade, double scale, bool calTime);
static void Draw(Mat& img, vector<Rect>& arrows, double scale);

static int arrowdetect_one_thread(bool useCPU, double scale, Mat image)
{
    CvCapture* capture = 0;

    ocl::OclCascadeClassifier cascade;
    
    cout<<"Started..."<<endl;

    if( !cascade.load( cascadeName ) )
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
    	detect(image, arrows, cascade, scale, i!=0);
    }
    cout << "done!" << endl;
    //cout << "average GPU time (noCamera) : ";
    //cout << getTotalTime() << " ms" << endl;
    cout << "Detected elements: "<<arrows.size()<<endl;

    Draw(image, arrows, scale);
    waitKey(0);
    
    cout<< "single-threaded sample has finished" <<endl;
    return 0;
}

// ---------------------------- NATIVE

JNIEXPORT void JNICALL Java_hr_foi_tiwo_ObjDec_sendImageForProcessing(JNIEnv *env, jclass cls, jlong image, jint width, jint height){
  
  cout<<"Hello from C++"<<endl;
  cascadeName = "haarcascade_10.xml";
  Mat* mat = (Mat*) image;
  arrowdetect_one_thread(true, 1.0, *mat);
    
}


// ---------------------------- END NATIVE

void detect( Mat& img, vector<Rect>& arrows, ocl::OclCascadeClassifier& cascade, double scale, bool calTime)
{
    ocl::oclMat image(img);
    ocl::oclMat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    
    if(calTime) workBegin();
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &before);


      ocl::cvtColor( image, gray, CV_BGR2GRAY );
      ocl::resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
      ocl::equalizeHist( smallImg, smallImg );
      cascade.detectMultiScale( smallImg, arrows, 1.1, 3, 0 | CV_HAAR_DO_ROUGH_SEARCH | CV_HAAR_DO_CANNY_PRUNING, Size(30,30), Size(0, 0) ); // 300ms
      // CV_HAAR_FIND_BIGGEST_OBJECT, CV_HAAR_DO_ROUGH_SEARCH, CV_HAAR_DO_CANNY_PRUNING, CV_HAAR_SCALE_IMAGE
      //cascade.detectMultiScale(cpu_smallImg, arrows, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30), Size(0, 0));


      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &after);

    if(calTime) workEnd();

    diff(before, after);

    
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
    imwrite("result-cpp-gpu.png", img);
}
