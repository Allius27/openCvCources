

double var_abs_laplacian(Mat image){
    ///
    /// YOUR CODE HERE
    
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    cv::Mat dst;
    
    cv::Laplacian(image, dst, CV_64F);

    cv::Scalar mu, sigma;
    cv::meanStdDev(dst, mu, sigma);

    double focusMeasure = sigma.val[0] * sigma.val[0];
    
    return focusMeasure;
    
    ///
}

double sum_modified_laplacian(Mat image){
    ///
    /// YOUR CODE HERE
    cv::Mat M = (Mat_<double>(3, 1) << -1, 2, -1);
    cv::Mat G = cv::getGaussianKernel(3, -1, CV_64F);

    cv::Mat Lx;
    cv::sepFilter2D(image, Lx, CV_64F, M, G);

    cv::Mat Ly;
    cv::sepFilter2D(image, Ly, CV_64F, G, M);

    cv::Mat FM = cv::abs(Lx) + cv::abs(Ly);

    double focusMeasure = cv::mean(FM).val[0];
    return focusMeasure;
    
    ///
}