#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main() {
    Mat source = imread("../data/sample.jpg");

    if (source.empty())
    {
        cout << "Unable to read file" << endl;
        return 2;
    }
    namedWindow("Window");

    // set up lambda callback
    cv::setMouseCallback(
         "Window",
         [] (int event, int x, int y, int flags, void* mat) {
             // top left and bottom right buttons
             static Point tl, rb;
             Mat image = *(Mat*)mat;

             if( event == EVENT_LBUTTONDOWN )
             {
                 tl.x = x;
                 tl.y = y;

                 rb.x = 0;
                 rb.y = 0;

                 circle(image, tl, 1, Scalar(0,255,0), 2 );
             }
             else if( event == EVENT_LBUTTONUP)
             {
                 rb.x = x;
                 rb.y = y;

                 circle(image, rb, 1, Scalar(0,255,0), 2 );
                 rectangle(image, tl, rb, Scalar(0,255,0), 2);

                 Rect rectRoi(tl.x, tl.y, rb.x - tl.x, rb.y - tl.y);
                 Mat dst;
                 image(rectRoi).copyTo(dst);

                 imwrite("cropped.jpg", dst);

                 tl.x = 0;
                 tl.y = 0;

                 putText(image,"Image have saved with name cropped.jpg" ,Point(10,90), FONT_HERSHEY_SIMPLEX, 0.7,Scalar(255,0,0), 2 );
            }
    }, &source );

    while(true)
    {
        imshow("Window", source );
        putText(source,"Choose top left corner and right bottom" ,Point(10,30), FONT_HERSHEY_SIMPLEX, 0.7,Scalar(255,0,0), 2 );
        putText(source,"Press ESC or q for exit" ,Point(10,60), FONT_HERSHEY_SIMPLEX, 0.7,Scalar(255,0,0), 2 );

        int k = waitKey(20) & 0xFF;
        if ( k == 27 || k == 'q' )
        {
            break;
        }
    }
    return 0;
}
