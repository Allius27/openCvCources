// Coin Detection Assignment
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void displayImage(Mat image){

//    resize(image,image,Size(), 0.3, 0.3);

	imshow("Image",image);
    if ( waitKey(0) == 'q' )
        exit(2);
//	destroyWindow(();
}

Mat displayConnectedComponents(Mat &im)
{
	// Make a copy of the image
	Mat imLabels = im.clone();

	// First let's find the min and max values in imLabels
	Point minLoc, maxLoc;
	double min, max;

	// The following line finds the min and max pixel values
	// and their locations in an image.
	minMaxLoc(imLabels, &min, &max, &minLoc, &maxLoc);

	// Normalize the image so the min value is 0 and max value is 255.
	imLabels = 255 * (imLabels - min) / (max - min);

	// Convert image to 8-bits
	imLabels.convertTo(imLabels, CV_8U);

	// Apply a color map
	Mat imColorMap;
	applyColorMap(imLabels, imColorMap, COLORMAP_JET);

	return imColorMap;
}

int main(){
    // Image path
    string imagePath = "../images/CoinsA.png";
	// Read image
	// Store it in the variable image
	///
	/// YOUR CODE HERE
	Mat image = imread( imagePath, IMREAD_COLOR);

    imshow("asd", image);

    if (image.empty())
        return EXIT_FAILURE;
	///

    displayImage(image);
	
	// Convert image to grayscale
	// Store it in the variable imageGray
	///
	/// YOUR CODE HERE
	Mat imageGray;
	cvtColor( image, imageGray, COLOR_BGR2GRAY );
	///
	
    displayImage(imageGray);
	
	// Split cell into channels
	// Store them in variables imageB, imageG, imageR
	///
	/// YOUR CODE HERE
    Mat bgr[3];
    split(image, bgr);

    Mat imageB = bgr[0];
    Mat imageG = bgr[1];
    Mat imageR = bgr[2];
    ///
	
//	displayImage(imageB);
//	displayImage(imageG);
//	displayImage(imageR);
	
	// Perform thresholding
	///
	/// YOUR CODE HERE
    Mat dst;
    threshold(imageGray, dst, 35, 255, cv::ThresholdTypes::THRESH_BINARY_INV);
	///

    // Modify as required
	// Perform morphological operations
	///
	/// YOUR CODE HERE
    Mat imageDilated = dst.clone();
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(10,10));
    int iterations = 3;

    for (int i=0; i<iterations; i++)
        dilate(imageDilated, imageDilated, kernel);
	///


	// Get structuring element/kernel which will be used for dilation
	///
	/// YOUR CODE HERE
    Mat imageEroded = imageDilated.clone();

    for (int i=0; i<iterations; i++)
        erode(imageEroded, imageEroded, kernel);
	///
	
    displayImage(imageEroded);
	
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	params.blobColor = 0;

	params.minDistBetweenBlobs = 2;

	// Filter by Area
	params.filterByArea = false;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.8;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.8;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.8;
	
	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	
	// Detect blobs
	///
	/// YOUR CODE HERE
    std::vector<KeyPoint> keypoints;
    detector->detect(imageEroded,keypoints);
	///
	
	// Print number of coins detected
	///
	/// YOUR CODE HERE
    cout << keypoints.size() << endl;
	///
	
	// Mark coins using image annotation concepts we have studied so far
	int x,y;
	int radius;
	double diameter;
	///
	/// YOUR CODE HERE

    for (int i=0; i< keypoints.size();i++)
    {
        int x = keypoints.at(i).pt.x;
        int y = keypoints.at(i).pt.y;
        int radius = keypoints.at(i).size/2;

        circle(image, Point(x, y), radius, Scalar(0,255,0),2);
    }

	///
	
    displayImage(image);
	
	// Find connected components
	// Use displayConnectedComponents function provided above
	///
	/// YOUR CODE HERE

    // Find connected components
    bitwise_not(imageEroded,imageEroded);
    morphologyEx(imageEroded,imageEroded,MORPH_OPEN, getStructuringElement(MORPH_CROSS, Size(3,3)));



	///
    displayImage(colorMap);
	
	// Find all contours in the image
	///
	/// YOUR CODE HERE
    Mat imLabels, colorMap;
    int nComponents = connectedComponents(imageEroded,imLabels);

    colorMap = displayConnectedComponents(imLabels);


    displayImage(imLabels);

    ///
	
	// Print the number of contours found
	///
	/// YOUR CODE HERE
    cout << "Number of connected components detected = " << nComponents << endl;;
    ///
	
	// Draw all contours
	///
	/// YOUR CODE HERE
    ///
	
	// Remove the inner contours
	// Display the result
	///
	/// YOUR CODE HERE

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(imageEroded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS,
                     cv::Point(-1, -1));

    std::cout << "Number of contours found = " << contours.size() << std::endl;

    ///
		
	// Print area and perimeter of all contours
	///
	/// YOUR CODE HERE

    std::vector<double> areas;

    for (int i=0; i < contours.size(); i++)
    {
        areas.push_back(contourArea(contours.at(i)));
        cout << "Contour #" << i
             << " has area = " << contourArea(contours.at(i))
             << " and perimeter = " << arcLength(contours.at(i), true)
             << endl;
    }


    cout << "max " << *max_element(areas.begin(), areas.end()) << endl;

    ///
	
	// Print maximum area of contour
	// This will be the box that we want to remove
	///
	/// YOUR CODE HERE
	///
	
	// Remove this contour and plot others
	///
	/// YOUR CODE HERE
	///
	
	// Fit circles on coins
	///
	/// YOUR CODE HERE
	///
	
    displayImage(image);
	
	// Image path
    imagePath = "../images/CoinsB.png";
	// Read image
	// Store it in variable image
	///
	/// YOUR CODE HERE
    if ( imagePath.empty() )
        return EXIT_FAILURE;

    image = imread(imagePath);

	// Convert image to grayscale
	// Store it in the variable imageGray
	///
	/// YOUR CODE HERE
    cvtColor(image, imageGray, COLOR_BGR2GRAY);

    ///
	
    displayImage(imageGray);
	
	// Split cell into channels
	// Store them in variables imageB, imageG, imageR
	///
	/// YOUR CODE HERE

    split(image, bgr);

    imageB = bgr[0];
    imageG = bgr[1];
    imageR = bgr[2];

    ///
	
//	displayImage(imageB);
//	displayImage(imageG);
//	displayImage(imageR);
	
	// Perform thresholding
	///
	/// YOUR CODE HERE

//    threshold(imageGray, dst, 140, 255 , THRESH_BINARY );

    threshold(imageGray, dst, 140, 100 , THRESH_BINARY_INV );
    ///
	
    displayImage(dst);
	
	// Perform morphological operations
	///
	/// YOUR CODE HERE

//    for (int i=0; i < 50; i++)
//        morphologyEx(dst,dst,MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(10,10)));

    for (int i=0; i < 50; i++)
        morphologyEx(dst,dst,MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(8,8)));


    displayImage(dst);

//    bitwise_not(dst ,dst);
    ///
	
	// Setup SimpleBlobDetector parameters.

	params.blobColor = 0;

	params.minDistBetweenBlobs = 2;

	// Filter by Area
	params.filterByArea = false;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.8;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.8;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.8;
	
	// Set up detector with params
	detector = SimpleBlobDetector::create(params);
	
	// Detect blobs
	///
    /// YOUR CODE HERE
    detector->detect(dst,keypoints);
	///
	
	// Print number of coins detected
	///
	/// YOUR CODE HERE
    cout << keypoints.size() << endl;
	///
	
	// Mark coins using image annotation concepts we have studied so far
	///
	/// YOUR CODE HERE

    for (KeyPoint p : keypoints)
            circle(image, p.pt, p.size/2,  Scalar(0,255,0),2);

    ///
	
	// Find connected components
	///
	/// YOUR CODE HERE
    nComponents = connectedComponents(dst, imLabels);
    cout << "Image B, amount of countors " << nComponents << endl;
    ///
	
	// Find all contours in the image
	///
	/// YOUR CODE HERE


    cv::findContours(dst, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS,
                     cv::Point(-1, -1));


    ///
	
	// Print the number of contours found
	///
	/// YOUR CODE HERE

    std::cout << "Image B; Number of contours found = " << contours.size() << std::endl;

    ///
	
	// Draw all contours
	///
	/// YOUR CODE HERE

    for (auto contour : contours)
        for (int i=1; i< contour.size(); i++)
        {
            Point p_current = contour.at(i);
            Point p_prev = contour.at(i-1);
            line(image, p_prev, p_current, Scalar(255,0,0),3);
        }

	///
	
	// Remove the inner contours
	// Display the result
	///
	/// YOUR CODE HERE


    ///
	
    displayImage(image);
	
	// Print area and perimeter of all contours
	///
	/// YOUR CODE HERE

    for (int i=0; i < contours.size(); i++)
        cout << "Contour #" << i
             << " has area = " << contourArea(contours.at(i))
             << " and perimeter = " << arcLength(contours.at(i), true)
             << endl;


    ///
	
	// Print maximum area of contour
	// This will be the box that we want to remove
	///
	/// YOUR CODE HERE
	///
	
	// Remove this contour and plot others
	///
	/// YOUR CODE HERE
	///
	
	// Print area and perimeter of all contours
	///
	/// YOUR CODE HERE
	///
	
	// Remove contours
	///
	/// YOUR CODE HERE
	///
	
	// Draw revised contours
	///
	/// YOUR CODE HERE
	///
	
	// Fit circles on coins
	///
	/// YOUR CODE HERE
	///
	
	displayImage(image);
	
	return 0;
}
