#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat cartoonify(Mat image, int arguments=0){

    Mat cartoonImage;

    /// YOUR CODE HERE

    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::Mat dst;
    cv::Laplacian(image, dst, CV_8U);
    dst.convertTo(dst, CV_8UC1);
    threshold(dst, dst, 30, 255, cv::ThresholdTypes::THRESH_BINARY);
    bitwise_not(dst, dst);
    erode(dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(2,2)));

    cartoonImage = dst.clone();
    cv::cvtColor(cartoonImage, cartoonImage, cv::COLOR_GRAY2BGR);

    return cartoonImage;
}

Mat pencilSketch(Mat image, int arguments=0){

    Mat pencilSketchImage;

    /// YOUR CODE HERE

    Mat cartoonMask = cartoonify(image);
    cvtColor (cartoonMask, cartoonMask, cv::COLOR_BGR2GRAY );
    bitwise_not(cartoonMask,cartoonMask);

    Mat cartoonMaskChannels[] = {cartoonMask,cartoonMask,cartoonMask};

    cartoonMask = cartoonMask/255;

    Mat imageChannels[3];
    Mat maskedImage[3];
    split(image, imageChannels);

    for (int i = 0; i < 3; i++)
    {
        // Use the mask to create the masked eye region
        multiply(imageChannels[i], (1-cartoonMaskChannels[i]), maskedImage[i]);
    }

    merge(maskedImage, 3, pencilSketchImage);

    return pencilSketchImage;
}

int main() {
    string imagePath = "../images/trump.jpg";
    Mat image = imread(imagePath);
    resize(image, image, Size(), 0.8, 0.8);
    Mat resultMat = Mat::zeros(image.size().height, image.size().width*3, CV_8UC3);

    Mat cartoonImage = cartoonify(image);
    Mat pencilSketchImage = pencilSketch(image);

    image.copyTo(resultMat(Range(0,image.size().height),Range(0,image.size().width)));
    cartoonImage.copyTo(resultMat(Range(0,image.size().height),Range(image.size().width,image.size().width*2)));
    pencilSketchImage.copyTo(resultMat(Range(0,image.size().height),Range(image.size().width*2,image.size().width*3)));

    imshow("image", resultMat);
    waitKey(0);
}
