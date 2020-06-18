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
    string imgPath = "images/IDCard-Satya.png";

    cv::Mat img = imread(imgPath);

    if ( img.empty() )
        return EXIT_FAILURE;

    Mat bbox, rectifiedImage;

    QRCodeDetector detector;
    std::string opencvData = detector.detectAndDecode(img, bbox, rectifiedImage);

    if(opencvData.length()>0)
        cout << "QR Code Detected" << endl;
    else
        cout << "QR Code NOT Detected" << endl;

    Point p1 =  Point(bbox.at<float>(0,0),bbox.at<float>(0,1));
    Point p2 =  Point(bbox.at<float>(0,2),bbox.at<float>(0,3));
    Point p3 =  Point(bbox.at<float>(0,4),bbox.at<float>(0,5));
    Point p4 =  Point(bbox.at<float>(0,6),bbox.at<float>(0,7));

    line(img, p1, p2,  Scalar(255,0,0), 2);
    line(img, p2, p3,  Scalar(255,0,0), 2);
    line(img, p3, p4,  Scalar(255,0,0), 2);
    line(img, p4, p1,  Scalar(255,0,0), 2);

    string resultImagePath = "./QRCode-Output.png";

    cout << imwrite(resultImagePath, rectifiedImage) << endl;





    return EXIT_SUCCESS;
}
