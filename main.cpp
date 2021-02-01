//01/02/2021
//Mikhail Zarechnev
//Test assignment


#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;

using namespace cv::xfeatures2d;

#define RATIO_THRESHOLD 0.74f
#define KNN_NEIGHB_NUM 2
#define GREEN Scalar(0, 255, 0)
#define LINE_THICK 4

#define _SHOW_DEBUG_OUTPUT_

int main( int argc, char* argv[] )
{
    if (argc != 3)
    {
        cout << "To execute application please provide required parameters" << endl;
        cout << "Format: ./testAssignment pathToCroppedImd pathToOrigImg" << endl;

        return -1;
     }

    Mat imgObject = imread(string(argv[1]), IMREAD_GRAYSCALE );
    //Mat img_object = imread(string(argv[2]),"/home/rapsodo/Downloads/Small_area_rotated.png", IMREAD_GRAYSCALE );

    Mat imgScene = imread(string(argv[2]), IMREAD_GRAYSCALE );
    //Mat imgScene = imread("/home/rapsodo/Downloads/StarMap.png", IMREAD_GRAYSCALE );

    //use SIFT for descriptor computation
    Ptr<SIFT> detector = SIFT::create();
    std::vector<KeyPoint> vecObjKeypoints, vecSceneKeypoints;
    Mat descriptorsObject, descriptorsScene;

    //compute descriptors and detect keypoints
    detector->detectAndCompute( imgScene, noArray(), vecSceneKeypoints, descriptorsScene);
    detector->detectAndCompute( imgObject, noArray(), vecObjKeypoints, descriptorsObject);

    //match descriptors
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > kNNMatches;

    matcher->knnMatch( descriptorsObject, descriptorsScene, kNNMatches, KNN_NEIGHB_NUM);

    //filter matches using the Lowe's ratio test (defaul = 0.75)
    std::vector<DMatch> vecGoodMatches;
    for (size_t i = 0; i < kNNMatches.size(); i++)
    {
       if (kNNMatches[i][0].distance < RATIO_THRESHOLD * kNNMatches[i][1].distance)
                vecGoodMatches.push_back(kNNMatches[i][0]);
    }

    //Compute object location on scene
    std::vector<Point2f> objKeypoints, sceneKeypoints;
    for( size_t i = 0; i < vecGoodMatches.size(); i++ )
    {
          objKeypoints.push_back(vecObjKeypoints[vecGoodMatches[i].queryIdx ].pt);
          sceneKeypoints.push_back(vecSceneKeypoints[vecGoodMatches[i].trainIdx ].pt);
    }
    Mat H = findHomography(objKeypoints, sceneKeypoints, RANSAC);
    //Obtain corners of detected area
    std::vector<Point2f> objCorners;
    objCorners.push_back(Point2f(0, 0));
    objCorners.push_back(Point2f((float)imgObject.cols, 0));
    objCorners.push_back(Point2f((float)imgObject.cols, (float)imgObject.rows));
    objCorners.push_back(Point2f(0, (float)imgObject.rows));

    std::vector<Point2f> sceneCorners;
    sceneCorners.resize(4);

    //apply homografy to object's corners
    perspectiveTransform( objCorners, sceneCorners, H);

    cout << "Detected corners:"  << endl;
    for(int i = 0; i < sceneCorners.size(); i++)
    {
        cout << i << ":" << sceneCorners[i] << endl;
    }

    Mat imgMatches;
    drawMatches( imgObject, vecObjKeypoints, imgScene, vecSceneKeypoints, vecGoodMatches, imgMatches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //draw detected area on scene
    line(imgMatches, sceneCorners[0] + Point2f((float)imgObject.cols, 0),
                  sceneCorners[1] + Point2f((float)imgObject.cols, 0), GREEN, LINE_THICK);
    line(imgMatches, sceneCorners[1] + Point2f((float)imgObject.cols, 0),
                  sceneCorners[2] + Point2f((float)imgObject.cols, 0), GREEN, LINE_THICK);
    line(imgMatches, sceneCorners[2] + Point2f((float)imgObject.cols, 0),
                  sceneCorners[3] + Point2f((float)imgObject.cols, 0), GREEN, LINE_THICK);
    line(imgMatches, sceneCorners[3] + Point2f((float)imgObject.cols, 0),
                  sceneCorners[0] + Point2f((float)imgObject.cols, 0), GREEN, LINE_THICK);

#ifdef _SHOW_DEBUG_OUTPUT_
     imshow("Matches And Detected Area", imgMatches );
     waitKey();
#endif
     imwrite("detectionResult.jpg", imgMatches);
     return 0;
}
