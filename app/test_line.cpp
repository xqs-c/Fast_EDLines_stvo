//输入一张低纹理图片，先提取点特征，显示结果；再同时提取点线特征，显示结果

#include <iostream>
#include "EDLib.h" //线特征提取器头文件
#include "ORBextractor.h"  //点特征提取器头文件
#include <line_descriptor_custom.hpp>
#include <line_descriptor/descriptor_custom.hpp>
#include <ctime>
#include <list>
#include <map>
using namespace cv;
using namespace std;
using namespace StVO;
using namespace line_descriptor;

//返回response大的线特征
struct sort_lines_by_response
{
    inline bool operator()(const KeyLine& a, const KeyLine& b){
        return ( a.response > b.response );
    }
};

struct sort_lines_by_length
{
    inline bool operator()(const KeyLine& a, const KeyLine& b){
        return ( a.lineLength > b.lineLength );
    }
};


int main(int argc, char **argv)
{	
    if(argc != 2){
        cout<<"usage: feature_extraction img"<<endl;
        return 1;
    }
        
    //************************** 输入图片 并显示源图像 **************************
    Mat testImg = imread(argv[1], IMREAD_GRAYSCALE);	
    assert(testImg.data != nullptr);
    imshow("Source Image", testImg);
    waitKey(0);
 

    //************************** 进行点特征提取，并显示结果 **************************
//     ORBextractor ORB(200,1.2,1,20,12); //创建一个ORB提取器
     vector<KeyPoint> points; //保存点特征的容器
     Mat pdesc; //保存描述子的容器
//     ORB.operator()(testImg,				//待提取特征点的图像
//                    cv::Mat(),		//掩摸图像, 实际没有用到
//                    points,			//输出变量，用于保存提取后的特征点
//                    pdesc);
    
     Ptr<ORB> orb = ORB::create( 100, 1.2, 1,
                                        19, 0, 2, 1,
                                        31, 80 );  //创建一个orb特征提取器
    orb->detectAndCompute( testImg, Mat(), points, pdesc, false);  //point特征点容器，pdesc特征点对应的描述子容器
    Mat pointimg;//输出图像
    //第一个参数image：原始图像，可以使三通道或单通道图像；
    //第二个参数keypoints：特征点向量，向量内每一个元素是一个KeyPoint对象，包含了特征点的各种属性信息；
    //第三个参数outImage：特征点绘制的画布图像，可以是原图像；
    //第四个参数color：绘制的特征点的颜色信息，默认绘制的是随机彩色；
    //第五个参数flags：特征点的绘制模式，其实就是设置特征点的那些信息需要绘制，那些不需要绘制
    drawKeypoints(testImg,points,pointimg,cv::Scalar(0,0,255),DrawMatchesFlags::DEFAULT);
    imshow("ORB feature",pointimg);
    imwrite("ORB_feature.png",pointimg);
    waitKey(0);
    
    
    //************************** 进行线特征提取，并显示结果 **************************
    EDPF testEDPF = EDPF(testImg); //无参边缘提取器
    EDLines testEDLines = EDLines(testEDPF);//线特征提取
    Mat lineImg = testEDLines.drawOnImage();	//draws on the input image
	imshow("Line Image", lineImg);
    //imwrite("LinesInImage_eu_1.png",lineImg);
    waitKey(0);
    
    drawKeypoints(lineImg,points,lineImg,cv::Scalar(0,0,255),DrawMatchesFlags::DRAW_OVER_OUTIMG);
    imshow("Point and Line Image", lineImg);
    waitKey(0);
    
     //************************** 取出提取到的线特征 **************************
	vector<LS> EDlines = testEDLines.getLines();
    
    //************************** 将提取到的线特征由LS格式转换为KeyLine格式 *************************
    unsigned int weight=testImg.cols;
    unsigned int height=testImg.rows;
    vector<KeyLine> lines;
    //遍历线特征对象转换为向量<KeyLine>
    lines.reserve(EDlines.size());
    for( int i = 0; i < EDlines.size(); i++ )
    {
        KeyLine kl;
        double octaveScale = 1.f;
        int    octaveIdx   = 0;

        kl.startPointX     = EDlines[i].start.x * octaveScale;
        kl.startPointY     = EDlines[i].start.y * octaveScale;
        kl.endPointX       = EDlines[i].end.x * octaveScale;
        kl.endPointY       = EDlines[i].end.y * octaveScale;

        kl.sPointInOctaveX = EDlines[i].start.x;
        kl.sPointInOctaveY = EDlines[i].start.y;
        kl.ePointInOctaveX = EDlines[i].end.x;
        kl.ePointInOctaveY = EDlines[i].end.y;

        kl.lineLength = (float) sqrt( pow( EDlines[i].start.x - EDlines[i].end.x, 2 ) + pow( EDlines[i].start.y - EDlines[i].end.y, 2 ) );

        kl.angle    = atan2( ( kl.endPointY - kl.startPointY ), ( kl.endPointX - kl.startPointX ) );
        kl.class_id = i;
        kl.octave   = octaveIdx;
        kl.size     = ( kl.endPointX - kl.startPointX ) * ( kl.endPointY - kl.startPointY );
        kl.pt       = Point2f( ( kl.endPointX + kl.startPointX ) / 2, ( kl.endPointY + kl.startPointY ) / 2 );

        kl.response = kl.lineLength / max( weight, height );
        cv::LineIterator li( testImg, Point2f( EDlines[i].start.x, EDlines[i].start.y ), Point2f( EDlines[i].end.x, EDlines[i].end.y ) );
        kl.numOfPixels = li.count;

        lines.push_back(kl);

    }
    
    //************************** 将提取到的KeyLine格式线特征进行筛选 *************************
    vector<KeyLine> v_lines;
    v_lines=lines;
    //筛选线特征
    //如果提取到的线特征数量大于线特征数量阈值，并且线特征数量阈值不等于0
    if( v_lines.size()>100 )
    {
        // sort lines by their response
        //根据线特征的response进行排序
        sort(v_lines.begin(),v_lines.end(), sort_lines_by_response() ); //按从大到小排列
        //lines.sort( v_lines.begin(),v_lines.end(), sort_lines_by_length() );  //按线特征的长度从大到小排列
        v_lines.resize(100);  //剔除多余的线特征
    }
    
    Mat Image_1 = Mat(testImg.size(), CV_8UC3);
	cvtColor(testImg, Image_1, COLOR_GRAY2BGR);
	for (vector<KeyLine>::iterator it =v_lines.begin();it!=v_lines.end();it++) {
        cv::Point startPoint= cv::Point(int(it->startPointX),int(it->startPointY));
		cv::Point endPoint   = cv::Point(int(it->endPointX),  int(it->endPointY));
		line(Image_1, startPoint, endPoint, Scalar(0, 255, 0), 1, LINE_AA, 0); // draw lines as green on image
        //line(colorImage, linePoints[i].start, linePoints[i].end, Scalar(0, 255, 0),1,LINE_AA);
	}
	imshow("200 Lines image",Image_1);
    waitKey(0);
    
    drawKeypoints(Image_1,points,Image_1,cv::Scalar(0,0,255),DrawMatchesFlags::DRAW_OVER_OUTIMG);
    imshow("Point and 200 Line Image", Image_1);
     imwrite("ORB_and_lines_feature.png",Image_1);
    waitKey(0);
    
    return 0;
}













