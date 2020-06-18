#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char * argv[])
{
    Mat demoImage = Mat::zeros(Size(10,10),CV_8U);

    demoImage.at<uchar>(0,1) = 1;
    demoImage.at<uchar>(9,0) = 1;
    demoImage.at<uchar>(8,9) = 1;
    demoImage.at<uchar>(2,2) = 1;
    demoImage(Range(5,8),Range(5,8)).setTo(1);

    Mat element = getStructuringElement(MORPH_CROSS, Size(3,3));

    int ksize = element.size().height;
    int height = demoImage.size().height;
    int width  = demoImage.size().width;



    //////////
    int border = ksize/2;
    Mat paddedDemoImage = Mat::zeros(Size(height + border*2, width + border*2),CV_8UC1);
    copyMakeBorder(demoImage,paddedDemoImage,border,border,border,border,BORDER_CONSTANT,0);

    Mat paddedDilatedImage = paddedDemoImage.clone();
    Mat mask;
    Mat resizedFrame;

    double minVal, maxVal;

    // Create a VideoWriter object
    // Use frame size as 50x50

//    float koeff = (float)50/(height + 2*border); int video = 50;
    float koeff = 50; int video = koeff * (height + border*2);

    VideoWriter outavi("output.avi",
                       cv::VideoWriter::fourcc('M','J','P','G'),
                       10, Size(video,video));

    Mat writeMat = paddedDemoImage.clone();

    for (int h_i = border; h_i < height + border; h_i++){
        for (int w_i = border; w_i < width + border; w_i++){
            {
                Mat smallImage, result;

                Rect roi (h_i-1, w_i-1, element.size().height, element.size().width);
                paddedDemoImage(roi).copyTo(smallImage);
                bitwise_and(smallImage, element, result);

                for (int r_h = 0; r_h < result.size().height; r_h++)
                {
                    for (int r_w = 0; r_w < result.size().width; r_w++)
                    {
                        cv::Mat diff;
                        cv::compare(result, element, diff, cv::CMP_NE);

                        if (cv::countNonZero(diff))
                            writeMat.at<uchar>(w_i, h_i) = 0;
                    }
                }
            }

            Mat output = Mat::zeros(Size(height + border*2, width + border*2),CV_8UC3);;
            resize(writeMat, output, Size(), koeff, koeff, INTER_NEAREST );
            cvtColor(output, output, COLOR_GRAY2RGB);

            output*=255;

            rectangle(output, Rect(h_i*koeff, w_i*koeff, koeff, koeff), (255,255,255),3);

            outavi.write(output);

            imshow("123", output);
            if ( waitKey() == 'q')
                return 0;
        }
    }

    outavi.release();

    return EXIT_SUCCESS;
}
