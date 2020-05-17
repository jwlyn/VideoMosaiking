//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <xfeatures2d.hpp>  
#include <opencv2/opencv.hpp>  
#include <cv.h>
#include <highgui.h>
#include<iomanip>
#include <iostream>  
#include <stdio.h>  
#include "gms_matcher.h" 
using namespace cv; 
using namespace xfeatures2d; 
using namespace std; 

#define MATCH_POINTS_SIZE_THRESH 30//The minimum number of feature points detected
#define SEAM_BELT_HALF_WIDTH	80//Feathering width
const double radian =  180.0/3.14159;


void people_mosaic(Mat &src, int mosaic_size)//Pedestrian detection + mosaic, mosaic_size is the size of the mosaic grid
{
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());// hog + SVM to detect pedestrians
	vector<Rect> found, found_filtered;
	hog.detectMultiScale(src, found, 0, Size(8,8), Size(64,64), 1.05, 2);
	 size_t i, j;
        for( i = 0; i < found.size(); i++ )
        {
            Rect r = found[i];
            for( j = 0; j < found.size(); j++ )
                if( j != i && (r & found[j]) == r)
                    break;
            if( j == found.size() )
                found_filtered.push_back(r);
        }
        for( i = 0; i < found_filtered.size(); i++ )
        {
            Rect r = found_filtered[i];
            r.x += cvRound(r.width*0.1);
            r.width = cvRound(r.width*0.8);
            r.y += cvRound(r.height*0.07);
            r.height = cvRound(r.height*0.8);

			Mat roi = src(r);
            int W=mosaic_size;
            int H=mosaic_size;

            //Mosaic process...                                                                                            
            for(int mi=W;mi<roi.cols;mi+=W)
                for(int mj=H;mj<roi.rows;mj+=H)
                {
					int b = roi.at<Vec3b>(mj-H/2,mi-W/2)[0];
					int g = roi.at<Vec3b>(mj-H/2,mi-W/2)[1];
					int r = roi.at<Vec3b>(mj-H/2,mi-W/2)[2];
                    for(int mx=mi-W;mx<=mi;mx++)
                        for(int my=mj-H;my<=mj;my++)
						{
							roi.at<Vec3b>(my,mx)[0] = b;
							roi.at<Vec3b>(my,mx)[1] = g;
							roi.at<Vec3b>(my,mx)[2] = r;
						}
                }
        }
}

//Define surf feature extractor, descriptor, matcher
Ptr<SURF> detector = SURF::create(100,4,3,0,0);
Ptr<DescriptorExtractor> descriptor = SURF::create();
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");

/** 
* @brief mosaicing   Function function: splicing the input image sequence to generate a spliced and cropped large image.
* @param img1             Input, new image to be stitched, RGB format
* @param img_map          Input, old big picture (ie the previous big picture that has been stitched together), RGB format
* @param img_map_result   Output, large image after splicing (for img_map of the next puzzle), RGB format
* @return                 Return value, 1 represents successful stitching, 0 represents failure (usually insufficient feature points or insufficient matching pairs) 
*/  
int  mosaicing(Mat &img1, Mat &img_map, Mat &img_map_result)
{
	int scale = 2;//The image is reduced by scale times, and feature point detection and matching are performed to increase the frame rate    
	bool isEnoughPoints = true; //Used to indicate whether there are enough feature points or matching pairs
	Rect roi_map;

	/************Calculate the approximate location of the area that matches img in img_map***********/
	int img2_w = img1.cols;
	int img2_h = img1.rows;
	roi_map.x = 0;
	roi_map.y = 0;
	roi_map.width = img2_w;
	roi_map.height = img2_h;	
	//img2 is intercepted from the old big picture img_map, used to match img1
	Mat img2;
	if(roi_map.width == img_map.cols && roi_map.height == img_map.rows)
		img_map.copyTo(img2);
	else	
		img2=img_map(roi_map);//img1

	Mat img1_bl,img2_bl;
	resize(img1,img1_bl,Size(img1.cols/scale,img1.rows/scale));//Zoom
	resize(img2,img2_bl,Size(img2.cols/scale,img2.rows/scale));//Zoom
	cv::GaussianBlur(img1_bl, img1_bl, cv::Size(5,5), 3, 3);//Gaussian blur denoising
	cv::GaussianBlur(img2_bl, img2_bl, cv::Size(5,5), 3, 3);//Gaussian blur denoising
	
	/****************Feature point detection***************/
	vector<KeyPoint> m_LeftKey,m_RightKey;  
	detector->detect( img1_bl, m_LeftKey );//Detect the feature points in img1 and store them in m_LeftKey
	detector->detect( img2_bl, m_RightKey );  //Detect the feature points in img2 and store them in m_RightKey
	if(m_LeftKey.size()<MATCH_POINTS_SIZE_THRESH*2 || m_RightKey.size()<MATCH_POINTS_SIZE_THRESH*2)
	{
		cout <<"m_LeftKey.size: " << m_LeftKey.size() <<" m_RightKey.size: " << m_RightKey.size()<< endl;
		isEnoughPoints = false;
		return 0;
	}
	
	Mat descriptors1,descriptors2;  
	vector<DMatch> matches;//Match result 
	vector<DMatch> goodMatches;  
	if(isEnoughPoints)
	{
		/****************Calculate the eigenvector matrix***************/ 
		descriptor->compute( img1_bl, m_LeftKey, descriptors1 );  
		descriptor->compute( img2_bl, m_RightKey, descriptors2 );  
		
		/****************Feature matching***************/ 
		matcher->match(descriptors1, descriptors2, matches);		
		//Calculate the maximum and minimum distance in the matching result
		double max_dist = 0;     
		double min_dist = 100; 
		for(int i=0; i<matches.size(); i++)  
		{  
			double dist = matches[i].distance;  
			if(dist < min_dist) min_dist = dist;  
			if(dist > max_dist) max_dist = dist;  
		}  		
		//Filter out better matching points
		for(int i=0; i<matches.size(); i++)  
		{  
			if(matches[i].distance < 8 * min_dist && matches[i].distance < 0.5 * max_dist)  
			{  
				goodMatches.push_back(matches[i]);  
			}  
		}  
	}
	else 
		return 0;

	/****************RANSAC filtering to get the homography matrix H***************/ 
	// Allocate space
	vector<DMatch> m_Matches=goodMatches;//goodMatches matches_gms
	int ptCount = (int)m_Matches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);
	// Convert Keypoint to Mat
	Point2f pt;
	for (int i=0; i<ptCount; i++)
	{
		pt = m_LeftKey[m_Matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x * scale;
		p1.at<float>(i, 1) = pt.y * scale;

		pt = m_RightKey[m_Matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x * scale;
		p2.at<float>(i, 1) = pt.y * scale;
	}
	if(ptCount < MATCH_POINTS_SIZE_THRESH*1.2)	
	{	
		isEnoughPoints = 0;
		return 0;
	}
	Mat H;
	if(isEnoughPoints)
	{
		// Calculate F with RANSAC method
		Mat m_Fundamental;
		vector<uchar> m_RANSACStatus;       // This variable is used to store the state of each point after RANSAC
		m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);// To solve the basic matrix, m_Fundamental is not actually needed, what is really needed is the matching status m_RANSACStatus of each pair of points. Filter and denoise by RANSAC method
		// Calculate the number of wild points
		int OutlinerCount = 0;
		for (int i=0; i<ptCount; i++)
		{
			if (m_RANSACStatus[i] == 0)    // The status is 0 means wild point
			{
				OutlinerCount++;
			}
		}
		int InlinerCount = ptCount - OutlinerCount;   // Calculate the interior point
		cout<<"Interior point："<<InlinerCount<<endl;
		if(InlinerCount < MATCH_POINTS_SIZE_THRESH)
		{	
			isEnoughPoints = 0;
			return 0;
		}
		else
		{
			vector<Point2f> m_LeftInlier;
			vector<Point2f> m_RightInlier;
			m_LeftInlier.resize(InlinerCount);
			m_RightInlier.resize(InlinerCount);
			InlinerCount=0;
			for (int i=0; i<ptCount; i++)
			{
				if (m_RANSACStatus[i] != 0)
				{
					m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
					m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
					m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
					m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
					InlinerCount++;
				}
			}
			//Perspective transformation, calculate the projection transformation matrix of img1 to img2. The matrix H is used to store the obtained projection matrix. Both LMEDS method and RANSAC method can be used
			H = findHomography( m_LeftInlier, m_RightInlier, LMEDS );//LMEDS RANSAC RHO
		}
	}

	/*****************Perspective transformation*****************/
	//Store the four corners of img1, and their positions transformed into img2
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0,0); obj_corners[1] = Point( img1.cols, 0 );
	obj_corners[2] = Point( img1.cols, img1.rows ); obj_corners[3] = Point( 0, img1.rows );
	std::vector<Point2f> scene_corners(4);
	if(isEnoughPoints)//If the number of points is sufficient, the perspective transformation is performed directly
	{
		perspectiveTransform( obj_corners, scene_corners, H);// Use the projection matrix H to get the mapped scene_corners coordinates
		cout<<scene_corners<<endl;
	}
	else 
		return 0;
		
	//Create a new image, imageturn, to store the img1 perspective transformed image. First calculate the length and width of the imageturn according to the coordinates of the four points of the projection point
	int width = int(max(abs(scene_corners[1].x), abs(scene_corners[2].x)));
	width =  int(max(width, img1.cols));
	int height = int(max(abs(scene_corners[2].y), abs(scene_corners[3].y)));
	height =  int(max(height, img1.rows));
	float origin_x=0,origin_y=0;
	if(scene_corners[0].x<0 ) 
	{
		if (scene_corners[3].x<0) 
			origin_x+=min(scene_corners[0].x,scene_corners[3].x);
		else 
			origin_x+=scene_corners[0].x;
	}
	else if(scene_corners[3].x<0 ) 
	{
		origin_x+=scene_corners[3].x;
	}	
	width-=int(origin_x);
	if(scene_corners[0].y<0) 
	{
		if (scene_corners[1].y<0) 
			origin_y+=min(scene_corners[0].y,scene_corners[1].y);
		else 
			origin_y+=scene_corners[0].y;
	}
	else if(scene_corners[1].y<0) 
	{
		origin_y+=scene_corners[1].y;
	}
	height-=int(origin_y);	
	Mat imageturn=Mat::zeros(height,width,img1.type());
	//Obtain a new transformation matrix so that the image is completely displayed
	for (int i=0;i<4;i++) {scene_corners[i].x -= origin_x; } 	
	for (int i=0;i<4;i++) {scene_corners[i].y -= origin_y; }
	Mat H1=getPerspectiveTransform(obj_corners, scene_corners);
	//Perform image transformation to get imageturn
	warpPerspective(img1,imageturn,H1,Size(width,height));	

	//After calculating the perspective transformation, the length of the top, bottom, left and right sides of the quadrilateral enclosed by the four vertices, and the aspect ratio parameters are used for subsequent judgments on the reliability of the transformation
	int up_length = (int)sqrt((scene_corners[1].y-scene_corners[0].y)*(scene_corners[1].y-scene_corners[0].y) + (scene_corners[1].x-scene_corners[0].x)*(scene_corners[1].x-scene_corners[0].x) );
	int down_length = (int)sqrt((scene_corners[2].y-scene_corners[3].y)*(scene_corners[2].y-scene_corners[3].y) + (scene_corners[2].x-scene_corners[3].x)*(scene_corners[2].x-scene_corners[3].x) );	
	int left_length = (int)sqrt((scene_corners[0].y-scene_corners[3].y)*(scene_corners[0].y-scene_corners[3].y) + (scene_corners[0].x-scene_corners[3].x)*(scene_corners[0].x-scene_corners[3].x) );
	int right_length = (int)sqrt((scene_corners[1].y-scene_corners[2].y)*(scene_corners[1].y-scene_corners[2].y) + (scene_corners[1].x-scene_corners[2].x)*(scene_corners[1].x-scene_corners[2].x) );	
	//The ratio of the left and right sides of the image
	float left_right_ratio = (float)min(left_length,right_length)/(float)max(left_length,right_length);
	cout << "up_length: "<<up_length << "down_length: "<< down_length << "left_length: "<< left_length << "right_length: "<< right_length <<endl;
	
	//Judging the convexity of the quadrilateral, isConv is 1 is convex
	vector<cv::Point >hull_src;
	hull_src.clear();
	hull_src.push_back(scene_corners[0]); hull_src.push_back(scene_corners[1]);
	hull_src.push_back(scene_corners[2]); hull_src.push_back(scene_corners[3]);
	bool isConv = isContourConvex(hull_src);

		
	/********Fine-adjust the four vertex positions of scene_corners, and update the imageturn after img1 perspective transformation. This place is the key to ensure the effect of long sequence image stitching********/
	if(  (fabs(scene_corners[1].y-scene_corners[0].y) > img1.rows*0.05  || left_right_ratio < 0.95 
	|| fabs((float)left_length - img1.rows) > img1.rows*0.1  || !isConv || !isEnoughPoints))
	{
		for (int i=0;i<4;i++) {scene_corners[i].x += origin_x; } 	
		for (int i=0;i<4;i++) {scene_corners[i].y += origin_y; }	
		float ratio = (float)img1.rows/(float)img1.cols;
		if(fabs(scene_corners[1].x - scene_corners[2].x)>5*2.5)//If the x coordinates of the two projection points on the right differ by more than the threshold, the x coordinates of scene_corners [1] and scene_corners [2] on the right are automatically adjusted
		{
			if(scene_corners[1].x > scene_corners[2].x){scene_corners[1].x-=fabs(scene_corners[1].x - scene_corners[2].x)/2;scene_corners[2].x+=fabs(scene_corners[1].x - scene_corners[2].x)/2;}
			if(scene_corners[1].x < scene_corners[2].x){scene_corners[1].x+=fabs(scene_corners[1].x - scene_corners[2].x)/2;scene_corners[2].x-=fabs(scene_corners[1].x - scene_corners[2].x)/2;}
		}
		if(right_length < img1.rows/1.05 || right_length > img1.rows*1.05)//If the right side is too long or too short, the y coordinate of the two projection points scene_corners [1] and scene_corners [2] on the right will be adjusted automatically.
		{
			if(((float)right_length < img1.rows/1.05))
			{
				float scene_corners_y_tmp = scene_corners[1].y;
				scene_corners[1].y = scene_corners[1].y - fabs(img1.rows - fabs(scene_corners[2].y - scene_corners[1].y))/2.0;
				scene_corners[2].y = scene_corners[2].y + fabs(img1.rows - fabs(scene_corners[2].y - scene_corners_y_tmp))/2.0;
			}
			else if(((float)right_length > img1.rows*1.05))
			{
				float scene_corners_y_tmp = scene_corners[1].y;
				scene_corners[1].y = scene_corners[1].y + fabs(img1.rows - fabs(scene_corners[2].y - scene_corners[1].y))/2.0;
				scene_corners[2].y = scene_corners[2].y - fabs(img1.rows - fabs(scene_corners[2].y - scene_corners_y_tmp))/2.0;
			}
		}
		//Update the coordinates of the projection points scene_corners [0] and scene_corners [3]. The constraint is that the length and width remain the same as img1 to avoid stretching and shrinking during the stitching process
		scene_corners[0].y = scene_corners[1].y;
		scene_corners[3].y = scene_corners[2].y;
		scene_corners[0].y -= fabs(img1.rows - fabs(scene_corners[3].y - scene_corners[0].y))/2.0;
		scene_corners[3].y += fabs(img1.rows - fabs(scene_corners[3].y - scene_corners[0].y))/2.0;
		scene_corners[0].x =  scene_corners[1].x - img1.cols;
		scene_corners[3].x =  scene_corners[2].x - img1.cols;
		//Update width and height for new imageturn
		width = int(max(abs(scene_corners[1].x), abs(scene_corners[2].x)));
		width =  int(max(width, img1.cols));
		height = int(max(abs(scene_corners[2].y), abs(scene_corners[3].y)));
		height =  int(max(height, img1.rows));
		origin_x=0;origin_y=0;
		if(scene_corners[0].x<0 ) 
		{
			if (scene_corners[3].x<0) 
				origin_x+=min(scene_corners[0].x,scene_corners[3].x);
			else 
				origin_x+=scene_corners[0].x;
		}
		else if(scene_corners[3].x<0 ) 
		{
			origin_x+=scene_corners[3].x;
		}	
		width-=int(origin_x);
		if(scene_corners[0].y<0) 
		{
			if (scene_corners[1].y<0) 
				origin_y+=min(scene_corners[0].y,scene_corners[1].y);
			else 
				origin_y+=scene_corners[0].y;
		}
		else if(scene_corners[1].y<0) 
		{
			origin_y+=scene_corners[1].y;
		}
		height-=int(origin_y);		
		imageturn.setTo(0);
		resize(imageturn,imageturn,Size(width,height));
		for (int i=0;i<4;i++) {scene_corners[i].x -= origin_x; } 	
		for (int i=0;i<4;i++) {scene_corners[i].y -= origin_y; }
		//Perform new projection transformation based on the new scene_corners
		H1=getPerspectiveTransform(obj_corners, scene_corners);
		//Generate new imageturn
		warpPerspective(img1,imageturn,H1,Size(width,height));	
	}

	/******Image stitching to generate imageturn after img1 and img2 are stitched together*******/
	uchar* ptr=imageturn.data;
	double alpha=0, beta=1;
	for (int row=0;row<height;row++) 
	{
		ptr=imageturn.data+row*imageturn.step+(int(0-origin_x))*imageturn.elemSize();
		if(row>=0- (int)origin_y&& row<img2.rows-(int)origin_y)
		{
			for(int col=0-(int)origin_x;col<img2.cols-(int)origin_x;col++)//for(int col=0;col<width_ol;col++)
			{
				uchar* ptr_c1=ptr+imageturn.elemSize1();  
				uchar* ptr_c2=ptr_c1+imageturn.elemSize1();
				uchar* ptr2=img2.data+(row + (int)origin_y)*img2.step+(col+(int)origin_x)*img2.elemSize();			
				uchar* ptr2_c1=ptr2+img2.elemSize1();  
				uchar* ptr2_c2=ptr2_c1+img2.elemSize1();

				if (*ptr==0&&*ptr_c1==0&&*ptr_c2==0) 
				{
					*ptr=(*ptr2);
					*ptr_c1=(*ptr2_c1);
					*ptr_c2=(*ptr2_c2);
				}			
			
				ptr+=imageturn.elemSize();
			}
		}		
	}

	//Image feathering, weighted average between the left and right sides of the imageturn and img2 image seams
	int min_brd = int(min(scene_corners[1].x,scene_corners[2].x) - SEAM_BELT_HALF_WIDTH);
	if (min_brd < int(-origin_x))
		min_brd = int(-origin_x);
	if (min_brd < 0)
		min_brd = 0;
	int max_brd = int(max(scene_corners[1].x, scene_corners[2].x) + SEAM_BELT_HALF_WIDTH);
	if (max_brd > img2.cols - (int)origin_x - 1)
		max_brd = img2.cols - (int)origin_x - 1;
	if (max_brd > imageturn.cols - 1)
		max_brd = imageturn.rows - 1;
	int mid_value = (min_brd + max_brd) / 2;
	cout <<"mid_value: "<<mid_value<<endl;
	alpha = 0.;double step_alpha = 1. / (max_brd - mid_value);
	int bg_col = max(int(max(-origin_y, scene_corners[1].y)),0);
	int end_col = min(int(min(scene_corners[2].y, img2.rows-origin_y)), imageturn.rows);
	for (int row = bg_col; row <= end_col; row++)
	{
		ptr = imageturn.data + row*imageturn.step + min_brd*imageturn.elemSize();
		for (int col = min_brd; col < max_brd; col++)
		{
			uchar* ptr_c1=ptr+imageturn.elemSize1();  
			uchar* ptr_c2=ptr_c1+imageturn.elemSize1();
			uchar* ptr2=img2.data+(row + (int)origin_y)*img2.step+(col+(int)origin_x)*img2.elemSize();			
			uchar* ptr2_c1=ptr2+img2.elemSize1();  
			uchar* ptr2_c2=ptr2_c1+img2.elemSize1();

			double beta = 1. - alpha;
			*ptr=(*ptr)*beta + (*ptr2)*alpha;
			*ptr_c1=(*ptr_c1)*beta + (*ptr2_c1)*alpha;
			*ptr_c2=(*ptr_c2)*beta + (*ptr2_c2)*alpha;

			ptr+=imageturn.elemSize();
			
			if (col <= mid_value)
				alpha += step_alpha;
			else 
				alpha -= step_alpha;
		}

		alpha = 0.;
	}	

	/******Generate a new big picture img_map_result_tmp and superimpose imageturn to img_map_result_tmp*******/
	int width_map = int(max(img_map.cols, roi_map.x + imageturn.cols + (int)origin_x));
	int height_map = int(max(img_map.rows, roi_map.y + imageturn.rows + (int)origin_y));
	float origin_x_map = 0,origin_y_map = 0;
	origin_x_map = (float)min(0,(int)(origin_x + roi_map.x));
	origin_y_map = (float)min(0,(int)(origin_y + roi_map.y));
	width_map -= origin_x_map;
	height_map -= origin_y_map;
	Mat img_map_result_tmp = Mat::zeros(height_map,width_map,img_map.type());
	uchar* ptr_map = img_map_result_tmp.data;	
	for (int row=0;row<height_map;row++) 
	{
		ptr_map = img_map_result_tmp.data+row*img_map_result_tmp.step+(int((roi_map.x+origin_x)-(int)origin_x_map))*img_map_result_tmp.elemSize();
		if(row>=(roi_map.y+origin_y)-(int)origin_y_map&& row<(roi_map.y+origin_y)-(int)origin_y_map + imageturn.rows-1 )
		{
			for(int col=roi_map.x+origin_x-(int)origin_x_map;col<imageturn.cols+roi_map.x+origin_x-(int)origin_x_map-1;col++)//for(int col=0;col<width_ol;col++)
			{
				uchar* ptr_c1=ptr_map+img_map_result_tmp.elemSize1();  
				uchar* ptr_c2=ptr_c1+img_map_result_tmp.elemSize1();
				uchar* ptr2=imageturn.data+(row - (int)((roi_map.y+origin_y)-(int)origin_y_map))*imageturn.step+(col-(int)(roi_map.x+origin_x-(int)origin_x_map))*imageturn.elemSize();
				uchar* ptr2_c1=ptr2+imageturn.elemSize1();  
				uchar* ptr2_c2=ptr2_c1+imageturn.elemSize1();

				if (*ptr_map==0&&*ptr_c1==0&&*ptr_c2==0) 
				{
					*ptr_map=(*ptr2);
					*ptr_c1=(*ptr2_c1);
					*ptr_c2=(*ptr2_c2);
				}			
				
				ptr_map+=img_map_result_tmp.elemSize();
			}
		}
	}

	/*******Overlay old big image img_map in new big image img_map_result_tmp******/
	uchar* ptr_map_ = img_map_result_tmp.data;
	for (int row=0;row<height_map;row++) 
	{
		ptr_map_ = img_map_result_tmp.data+row*img_map_result_tmp.step+(int(0-origin_x_map))*img_map_result_tmp.elemSize();
		if(row>=0- (int)origin_y_map&& row<img_map.rows-(int)origin_y_map-1)
		{
			for(int col=0-(int)origin_x_map;col<img_map.cols-(int)origin_x_map-1;col++)//for(int col=0;col<width_ol;col++)
			{
				uchar* ptr_c1=ptr_map_+img_map_result_tmp.elemSize1();  
				uchar* ptr_c2=ptr_c1+img_map_result_tmp.elemSize1();
				uchar* ptr2=img_map.data+(row + (int)origin_y_map)*img_map.step+(col+(int)origin_x_map)*img_map.elemSize();
				uchar* ptr2_c1=ptr2+img_map.elemSize1();  
				uchar* ptr2_c2=ptr2_c1+img_map.elemSize1();
			
				if (*ptr_map_==0&&*ptr_c1==0&&*ptr_c2==0) 
				{
					*ptr_map_=(*ptr2);
					*ptr_c1=(*ptr2_c1);
					*ptr_c2=(*ptr2_c2);
				}
				ptr_map_+=img_map_result_tmp.elemSize();
			}
		}
	}


	img_map_result_tmp.copyTo(img_map_result);
	Mat rsz;
	resize(img_map_result,rsz,Size(img_map_result.cols/4,img_map_result.rows/4));
	imshow("img_map_result_tmp1",rsz);
	return 1;
}

int main()
{
	const std::string path_images="/input"; 

	Mat img_map_result;
	cout.precision(8);
	int begin_index = 100;
	int end_index = 118;
	
	for(int i = begin_index;i <= end_index;i++)
	{ 
	
		cout << "begin the "<< i << "th image!"<<endl;
		std::string matchName = path_images + std::to_string((long long)(i)) + ".bmp"; 	 
		Mat map_tmp = img_map_result;
		if(i == begin_index)/
			continue;
		if(i == begin_index + 1)
		{
			 std::string matchName_img2 = path_images + std::to_string((long long)(begin_index)) + ".bmp"; 
			 Mat img2 = imread(matchName_img2);  
			 people_mosaic(img2, 8);
			 img2.copyTo(map_tmp);
	    }
		Mat img_ori = imread(matchName);
		Mat img;
		people_mosaic(img_ori, 8);
		img_ori.copyTo(img);
		mosaicing(img, map_tmp, img_map_result);		
		waitKey(0);
		cout << "end the "<< i << "th image!"<<endl;
	}

	imshow("image_overlap", img_map_result);
	imwrite("result.jpg",img_map_result);

	waitKey(0);
	return 0;
}


