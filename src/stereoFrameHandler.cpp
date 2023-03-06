/*****************************************************************************
**      Stereo VO and SLAM by combining point and line segment features     **
******************************************************************************
**                                                                          **
**  Copyright(c) 2016-2018, Ruben Gomez-Ojeda, University of Malaga         **
**  Copyright(c) 2016-2018, David Zuñiga-Noël, University of Malaga         **
**  Copyright(c) 2016-2018, MAPIR group, University of Malaga               **
**                                                                          **
**  This program is free software: you can redistribute it and/or modify    **
**  it under the terms of the GNU General Public License (version 3) as     **
**  published by the Free Software Foundation.                              **
**                                                                          **
**  This program is distributed in the hope that it will be useful, but     **
**  WITHOUT ANY WARRANTY; without even the implied warranty of              **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            **
**  GNU General Public License for more details.                            **
**                                                                          **
**  You should have received a copy of the GNU General Public License       **
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.   **
**                                                                          **
*****************************************************************************/

#include <stereoFrameHandler.h>  //双目图像处理头文件，即跟踪线程

#include "matching.h"
#include <random>

namespace StVO{
//默认构造函数
StereoFrameHandler::StereoFrameHandler( PinholeStereoCamera *cam_ ) : cam(cam_)
{
    //设置ORB特征提取器的参数
    // 每一帧提取的特征点数 1000
    int nFeatures = Config::orbNFeatures();
    // 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = Config::orbScaleFactor();
    // 尺度金字塔的层数 8
    int nLevels = Config::orbNLevels();
    // 提取fast特征点的默认阈值 20
    int fIniThFAST = Config::orbFastTh();
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    int fMinThFAST = Config::orbFastminTh();
    // tracking过程都会用到mpORBextractorLeft作为特征点提取器
    mpORBextractorLeft = new ORBextractor(
        nFeatures,      //参数的含义还是看上面的注释吧
        fScaleFactor,
        nLevels,
        fIniThFAST,
        fMinThFAST);
    mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
}

StereoFrameHandler::~StereoFrameHandler()
{
    delete mpORBextractorLeft;
    delete mpORBextractorRight;
}

/*  main methods  */
//双目初始化
//img_l_：左目图像  img_r_：右目图像  idx_：图像帧的索引
void StereoFrameHandler::initialize(const Mat img_l_, const Mat img_r_ , const int idx_)
{
    // variables for adaptative thresholds
    orb_fast_th = Config::orbFastTh();//配置文件中orb特征提取的阈值
    llength_th  = Config::minLineLength() * std::min( cam->getWidth(), cam->getHeight() ) ; //线段的最小长度阈值，使用相机长、宽中的较小值*阈值
    // define StereoFrame右
    prev_frame = new StereoFrame( img_l_, img_r_, idx_, cam ,mpORBextractorLeft,mpORBextractorRight,0.0,0.0); //创建前一帧,第一帧的位姿优化误差为0
    prev_frame->extractStereoFeatures( llength_th, orb_fast_th );  //提取双目点线特征，并完成双目点线特征匹配和点线特征的三角化
    prev_frame->Tfw     = Matrix4d::Identity();  //初始化变换矩阵
    prev_frame->Tfw_cov = Matrix6d::Identity();  //初始化变换矩阵的协方差
    prev_frame->DT      = Matrix4d::Identity();
    curr_frame = prev_frame;  //将前一帧赋值给当前帧
    // SLAM variables for KF decision
    T_prevKF         = Matrix4d::Identity();  //初始化前一帧到当前帧的转移矩阵
    cov_prevKF_currF = Matrix6d::Zero();   //初始化前一帧到当前帧的转移矩阵的协方差矩阵
    prev_f_iskf      = true; //将前一帧是关键帧的标志位置为true 
    N_prevKF_currF   = 0;  //前一关键帧到当前帧间隔的帧数为0
    
}

//从数据集类中读取双目图像对，并进行处理
void StereoFrameHandler::insertStereoPair(const Mat img_l_, const Mat img_r_ , const int idx_)
{
    //curr_frame.reset( new StereoFrame( img_l_, img_r_, idx_, cam ) );
    //创建当前帧 ,并将前一帧的位姿优化误差传递给当前帧的前一帧位姿优化误差，前一帧的前一帧位姿优化误差传递给当前帧的前前帧位姿优化误差，实现帧关联
    curr_frame =  new StereoFrame( img_l_, img_r_, idx_, cam ,mpORBextractorLeft,mpORBextractorRight,prev_frame->err_norm,prev_frame->prev_frame_err_norm);  
    curr_frame->extractStereoFeatures( llength_th, orb_fast_th );//提取双目点线特征，并完成双目点线特征匹配和点线特征的三角化
    f2fTracking();  //开启跟踪线程
}

void StereoFrameHandler::updateFrame()
{

    // update FAST threshold for the keypoint detection
    //更新关键点检测的快速阈值
    if( Config::adaptativeFAST() )
    {
        int min_fast  = Config::fastMinTh();
        int max_fast  = Config::fastMaxTh();
        int fast_inc  = Config::fastIncTh();
        int feat_th   = Config::fastFeatTh();
        float err_th  = Config::fastErrTh();

        // if bad optimization, -= 2*fast_inc
        if( curr_frame->DT == Matrix4d::Identity() || curr_frame->err_norm > err_th )
            orb_fast_th = std::max( min_fast, orb_fast_th - 2*fast_inc );
        // elif number of features ...
        else if( n_inliers_pt < feat_th )
            orb_fast_th = std::max( min_fast, orb_fast_th - 2*fast_inc );
        else if( n_inliers_pt < feat_th * 2 )
            orb_fast_th = std::max( min_fast, orb_fast_th - fast_inc );
        else if( n_inliers_pt > feat_th * 3 )
            orb_fast_th = std::min( max_fast, orb_fast_th + fast_inc );
        else if( n_inliers_pt > feat_th * 4 )
            orb_fast_th = std::min( max_fast, orb_fast_th + 2*fast_inc );
    }

    // clean and update variables
    for( auto pt: matched_pt )
        delete pt;
    for( auto ls: matched_ls )
        delete ls;
    matched_pt.clear();
    matched_ls.clear();

    //prev_frame.reset();
    //prev_frame.reset( curr_frame );器
    delete prev_frame;
    prev_frame = curr_frame;
    curr_frame = NULL;

}

/*  tracking methods  */
//跟踪双目帧
void StereoFrameHandler::f2fTracking()
{

    // feature matching
    //特征匹配
    matched_pt.clear(); //清空与上一帧的点特征匹配容器
    matched_ls.clear();  //清空与上一帧的线特征匹配容器

    //如果点线并行处理标志位、提取点特征和提取线特征标志位的为1
    if( Config::plInParallel() && Config::hasPoints() && Config::hasLines() )
    {
        auto detect_p = async(launch::async, &StereoFrameHandler::matchF2FPoints, this );  //创建一个线程进行帧间点特征匹配
        auto detect_l = async(launch::async, &StereoFrameHandler::matchF2FLines,  this );  //创建一个线程进行帧间线特征匹配
        detect_p.wait();
        detect_l.wait();
    }
    else
    {
        if (Config::hasPoints()) matchF2FPoints();
        if (Config::hasLines()) matchF2FLines();
    }

    n_inliers_pt = matched_pt.size(); //正确匹配的内点数量
    n_inliers_ls = matched_ls.size();//正确匹配的内线数量
    n_inliers    = n_inliers_pt + n_inliers_ls;  //正确匹配的特征数量（点+线）
}


//点特征的帧间匹配
void StereoFrameHandler::matchF2FPoints()
{

    // points f2f tracking
    // --------------------------------------------------------------------------------------------------------------------
    matched_pt.clear();  //清空点特征匹配容器 存放与当前帧匹配的上一帧点特征
    if ( !Config::hasPoints() || curr_frame->stereo_pt.empty() || prev_frame->stereo_pt.empty() )
        return; //如果没有提取点特征、当前帧的双目点特征容器为空或者上一帧的双目点特征为空，则返回

    std::vector<int> matches_12;  //创建一个帧间匹配对容器，存放当前帧匹配的特征点的索引
    //进行帧间匹配
    //输入参数：上一帧的左目特征点的描述子 当前帧的左目特征点的描述子  最佳距离和次佳距离比值的阈值  匹配对容器
    //结果存放在匹配对容器中
    match(prev_frame->pdesc_l, curr_frame->pdesc_l, Config::minRatio12P(), matches_12);

    // bucle around pmatches
    //遍历每一对匹配点对
    for (int i1 = 0; i1 < matches_12.size(); ++i1) {
        const int i2 = matches_12[i1];//取出匹配的当前帧特征点的索引
        if (i2 < 0) continue;  //如果匹配的特征点的索引非法，则跳过这对匹配对

        prev_frame->stereo_pt[i1]->pl_obs = curr_frame->stereo_pt[i2]->pl;  //将当前帧的特征点赋值给匹配的前一帧特征点的投影点
        prev_frame->stereo_pt[i1]->inlier = true;  //将上一帧特征点是内点的标志位置为true
        matched_pt.push_back( prev_frame->stereo_pt[i1]->safeCopy() );  //将前一帧的特征点深拷贝到匹配容器中
        curr_frame->stereo_pt[i2]->idx = prev_frame->stereo_pt[i1]->idx; // prev idx  将前一帧匹配的特征点的索引赋值给当前帧匹配的特征点的索引
    }
}

//线特征的帧间匹配
void StereoFrameHandler::matchF2FLines()
{

    // line segments f2f tracking
    matched_ls.clear(); //清空线特征匹配对容器，存放当前帧线特征匹配的上一帧线特征
    if( !Config::hasLines() || curr_frame->stereo_ls.empty() || prev_frame->stereo_ls.empty() )
        return;

    std::vector<int> matches_12;  //创建一个帧间匹配对容器
    //进行线特征匹配
    //输入：前一帧左目线特征描述子  当前帧左目线特征描述子  最佳距离和次佳距离比值的阈值 匹配对容器
    match(prev_frame->ldesc_l, curr_frame->ldesc_l, Config::minRatio12L(), matches_12);

    vector<KeyLine> lines_c=curr_frame->lines_l; //当前帧左目线特征
    vector<KeyLine> lines_p=prev_frame->lines_l;  //上一帧左目线特征
    //这里可以加入线特征匹配的几何约束
    //遍历每一对线匹配对，检查是否满足线段匹配的约束
    for (int i1 = 0; i1 < matches_12.size(); ++i1) {
        const int i2 = matches_12[i1];  //取出与当前帧线特征相匹配的上一帧线特征的索引
        if (i2 < 0) continue;  //如果索引小于0，说明未成功匹配，跳过该左目线特征
/*
        string angPath="angle.txt";
        string lenPath="len.txt";
        ofstream of,ofl;
        of.open(angPath.c_str(),ios::out | ios::app);
        ofl.open(lenPath.c_str(),ios::out | ios::app);*/
        
        //角度滤波
        double ang_c=lines_c[i1].angle;
        double ang_p=lines_p[i2].angle;
        double ang=abs(abs(ang_c)-abs(ang_p));
        
        //长度滤波
        double len_c=lines_c[i1].lineLength;
        double len_p=lines_p[i2].lineLength;
        double len=min(len_c,len_p)/max(len_c,len_p);
        
//         of<<ang<<endl;
//         ofl<<len<<endl;
//         of.close();
//         ofl.close();
        
        if(1)
        {
            if (ang > 0.2 || len<0.8)
            {
                matches_12[i1]=-1;
            }
        }
   }
    
    // bucle around pmatches
    //遍历每一对匹配对
    for (int i1 = 0; i1 < matches_12.size(); ++i1) {
        const int i2 = matches_12[i1];  //取出匹配的当前帧线特征索引
        if (i2 < 0) continue; //如果索引非法，跳过该匹配对

        prev_frame->stereo_ls[i1]->sdisp_obs = curr_frame->stereo_ls[i2]->sdisp;  //
        prev_frame->stereo_ls[i1]->edisp_obs = curr_frame->stereo_ls[i2]->edisp;  //
        prev_frame->stereo_ls[i1]->spl_obs   = curr_frame->stereo_ls[i2]->spl;  //
        prev_frame->stereo_ls[i1]->epl_obs   = curr_frame->stereo_ls[i2]->epl;  //
        prev_frame->stereo_ls[i1]->le_obs    = curr_frame->stereo_ls[i2]->le;  //
        prev_frame->stereo_ls[i1]->inlier    = true;  //将该线特征是内线的标志位置true
        matched_ls.push_back( prev_frame->stereo_ls[i1]->safeCopy() );  //深拷贝上一帧线特征存放到匹配容器中
        curr_frame->stereo_ls[i2]->idx = prev_frame->stereo_ls[i1]->idx; // prev idx
    }
}

double StereoFrameHandler::f2fLineSegmentOverlap( Vector2d spl_obs, Vector2d epl_obs, Vector2d spl_proj, Vector2d epl_proj  )
{

    double overlap = 1.f;

    if( std::abs(spl_obs(0)-epl_obs(0)) < 1.0 )         // vertical lines
    {

        // line equations
        Vector2d l = epl_obs - spl_obs;

        // intersection points
        Vector2d spl_proj_line, epl_proj_line;
        spl_proj_line << spl_obs(0), spl_proj(1);
        epl_proj_line << epl_obs(0), epl_proj(1);

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(1)-spl_obs(1)) / l(1);
        double lambda_e = (epl_proj_line(1)-spl_obs(1)) / l(1);

        double lambda_min = min(lambda_s,lambda_e);
        double lambda_max = max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }
    else if( std::abs(spl_obs(1)-epl_obs(1)) < 1.0 )    // horizontal lines (previously removed)
    {

        // line equations
        Vector2d l = epl_obs - spl_obs;

        // intersection points
        Vector2d spl_proj_line, epl_proj_line;
        spl_proj_line << spl_proj(0), spl_obs(1);
        epl_proj_line << epl_proj(0), epl_obs(1);

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(0)-spl_obs(0)) / l(0);
        double lambda_e = (epl_proj_line(0)-spl_obs(0)) / l(0);

        double lambda_min = min(lambda_s,lambda_e);
        double lambda_max = max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }
    else                                            // non-degenerate cases
    {

        // line equations
        Vector2d l = epl_obs - spl_obs;
        double a = spl_obs(1)-epl_obs(1);
        double b = epl_obs(0)-spl_obs(0);
        double c = spl_obs(0)*epl_obs(1) - epl_obs(0)*spl_obs(1);

        // intersection points
        Vector2d spl_proj_line, epl_proj_line;
        double lxy = 1.f / (a*a+b*b);

        spl_proj_line << ( b*( b*spl_proj(0)-a*spl_proj(1))-a*c ) * lxy,
                         ( a*(-b*spl_proj(0)+a*spl_proj(1))-b*c ) * lxy;

        epl_proj_line << ( b*( b*epl_proj(0)-a*epl_proj(1))-a*c ) * lxy,
                         ( a*(-b*epl_proj(0)+a*epl_proj(1))-b*c ) * lxy;

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(0)-spl_obs(0)) / l(0);
        double lambda_e = (epl_proj_line(0)-spl_obs(0)) / l(0);

        double lambda_min = min(lambda_s,lambda_e);
        double lambda_max = max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }

    return overlap;

}

/*  optimization functions */

bool StereoFrameHandler::isGoodSolution( Matrix4d DT, Matrix6d DTcov, double err )
{
    SelfAdjointEigenSolver<Matrix6d> eigensolver(DTcov);
    Vector6d DT_cov_eig = eigensolver.eigenvalues();  //求取变化矩阵的协方差矩阵的特征值
    //如果协方差矩阵的第一个特征值小于0，或者最后一个特征值大于1，后者误差小于0，或者误差大于1，或者变化矩阵不是
    if( DT_cov_eig(0)<0.0 || DT_cov_eig(5)>1.0 || err < 0.0 || err > 1.0 || !is_finite(DT) )
    {
        cout << endl << DT_cov_eig(0) << "\t" << DT_cov_eig(5) << "\t" << err <<"  不是好的位姿优化"<< endl;
        return false; //显示错误信息，并返回false
    }

    return true;
}


//进行位姿优化
void StereoFrameHandler::optimizePose()
{

    // definitions
    Matrix4d DT, DT_; //定义从前一帧到当前帧位姿变化矩阵
    Matrix6d DT_cov; //位姿变化矩阵的协方差矩阵
    double   err = numeric_limits<double>::max(), e_prev;
    err = -1.0;
    
//     string saveError = "/home/xqs/PL-SLAM/stvo-pl-1.0/error.txt"; //保存位姿优化后投影误差的文件的路径
//     ofstream ofe;
//     ofe.open(saveError.c_str(),ios::out | ios::app);
//     ofe<<prev_frame->err_norm<<endl;
//     ofe.close();
    //cout<<endl<<"上一帧位姿优化的误差： "<<prev_frame->err_norm<<endl;//如果上一帧位姿误差过大或者上一帧跟踪失败，则不应使用匀速模型
    

//     string fPath="f_y.txt";
//     ofstream of;
//     of.open(fPath.c_str(),ios::out | ios::app);
    
    double f_y=0.0;
    if(prev_frame->err_norm == -1 || prev_frame->prev_frame_err_norm == -1)
    {
        f_y=1;
    }
    else
    {
        f_y=0.6*prev_frame->err_norm+0.4*prev_frame->prev_frame_err_norm;
    }
    
//     of<<f_y<<endl;
//     of.close();
    
    // set init pose (depending on the use of prior information or not, and on the goodness of previous solution)
    //设置初始姿态（取决于是否使用先验信息，以及先前解决方案的优点）
    if( Config::useMotionModel() && f_y<=0.045 && prev_frame->err_norm != -1) //如果使用匀速运动模型标志位置1 
    {
        DT     = prev_frame->DT; //将前一帧的位姿变化矩阵赋值给当前帧
        DT_cov = prev_frame->DT_cov;  //将前一帧位姿变化矩阵的协方差矩阵赋值给当前帧
        e_prev = prev_frame->err_norm;
        if( !isGoodSolution(DT,DT_cov,e_prev) )
            DT = Matrix4d::Identity();  //判断是否是好的，如果不是，则初始化变化矩阵为单位矩阵
    }
    else
        DT = Matrix4d::Identity(); //如果不使用匀速运动模型，则初始化变化矩阵为单位矩阵

    // optimization mode
    int mode = 1;   // GN - GNR - LM 优化器模型，高斯、列文伯格-马夸尔特

    // solver
    //如果匹配的点线特征大于最小特征数的阈值，则进行优化和细优化
    if( n_inliers >= Config::minFeatures() )
    {
        // optimize
        DT_ = DT; //创建一个位姿的副本
        if( mode == 0 )      gaussNewtonOptimization(DT_,DT_cov,err,Config::maxIters());  //高斯优化  Config::maxIters()优化第一阶段的迭代次数 默认为5
        else if( mode == 1 ) gaussNewtonOptimizationRobust(DT_,DT_cov,err,Config::maxIters());  //带鲁棒核的高斯优化
        else if( mode == 2 ) levenbergMarquardtOptimization(DT_,DT_cov,err,Config::maxIters());//列文伯格-马夸尔特优化

        // remove outliers (implement some logic based on the covariance's eigenvalues and optim error)
        //去除异常值（根据协方差的特征值和最优误差实现一些逻辑）
        if( isGoodSolution(DT_,DT_cov,err) )
        {
            removeOutliers(DT_); //去除异常点
            // refine without outliers
            //去除异常值后特征数量依旧大于最小特征数量阈值
            if( n_inliers >= Config::minFeatures() )
            {
                //在进行一次优化，细优化
                if( mode == 0 )      gaussNewtonOptimization(DT,DT_cov,err,Config::maxItersRef());// Config::maxItersRef()细化阶段中的迭代次数 默认为10
                else if( mode == 1 ) gaussNewtonOptimizationRobust(DT,DT_cov,err,Config::maxItersRef());
                else if( mode == 2 ) levenbergMarquardtOptimization(DT,DT_cov,err,Config::maxItersRef());
            }
            else
            {
                //如果剔除异常值后特征数量不足，将位姿转移矩阵初始化为单位矩阵，输出错误信息
                DT     = Matrix4d::Identity();
                cout << "[StVO] not enough inliers (after removal)" << endl;
            }
        }
        else
        {
            //如果不是好的结果，采用带鲁棒核的高斯优化
            gaussNewtonOptimizationRobust(DT,DT_cov,err,Config::maxItersRef());
            //DT     = Matrix4d::Identity();
            //cout << "[StVO] optimization didn't converge" << endl;
        }
    }
    else
    {
        //如果提取的特征数量小于阈值，输出错误信息
        DT     = Matrix4d::Identity();
        cout << "[StVO] not enough inliers (before optimization)" << endl;
    }


    // set estimated pose
    //如果结果是好的
    if( isGoodSolution(DT,DT_cov,err) && DT != Matrix4d::Identity() )
    {
        curr_frame->DT       = expmap_se3(logmap_se3( inverse_se3( DT ) ));//保存优化后位姿
        curr_frame->DT_cov   = DT_cov; //保存位姿矩阵的协方差
        curr_frame->err_norm = err; //保存投影误差的平均值
        curr_frame->Tfw      = expmap_se3(logmap_se3( prev_frame->Tfw * curr_frame->DT )); //保存当前帧世界坐标->相机坐标的转移矩阵
        curr_frame->Tfw_cov  = unccomp_se3( prev_frame->Tfw, prev_frame->Tfw_cov, DT_cov );//保存转移矩阵的协方差
        SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
        curr_frame->DT_cov_eig = eigensolver.eigenvalues(); //保存传递值
    }
    else
    {
        //位姿优化失败，初始化值
        //setAsOutliers();
        curr_frame->DT       = Matrix4d::Identity();
        curr_frame->DT_cov   = Matrix6d::Zero();
        curr_frame->err_norm = -1.0;
        curr_frame->Tfw      = prev_frame->Tfw;  //将当前帧的位姿赋值为上一帧的位姿
        curr_frame->Tfw_cov  = prev_frame->Tfw_cov;
        curr_frame->DT_cov_eig = Vector6d::Zero();
    }
}


//利用高斯牛顿法进行位姿优化
//输入：DT 上一帧到当前帧的转移矩阵 DT_cov：转移矩阵的协方差  err_：投影误差  max_iters：迭代次数
void StereoFrameHandler::gaussNewtonOptimization(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters)
{
    Matrix6d H;
    Vector6d g, DT_inc;
    double err, err_prev = 999999999.9;

    int iters;
    //进行迭代
    for( iters = 0; iters < max_iters; iters++)
    {
        // estimate hessian and gradient (select)
        //估计hessian矩阵和梯度（选择）
        optimizeFunctions( DT, H, g, err );
        if (err > err_prev) {
            if (iters > 0)
                break;
            err_ = -1.0;
            return;
        }
        // if the difference is very small stop
        //如果差异很小，停止 
        if( ( ( err < Config::minError()) || abs(err-err_prev) < Config::minErrorChange() ) ) {
            cout << "[StVO] Small optimization improvement" << endl;
            break;
        }
        // update step
        ColPivHouseholderQR<Matrix6d> solver(H);
        DT_inc = solver.solve(g);
        DT  << DT * inverse_se3( expmap_se3(DT_inc) );
        // if the parameter change is small stop
        //如果参数变化较小，则停止 
        if( DT_inc.head(3).norm() < Config::minErrorChange() && DT_inc.tail(3).norm() < Config::minErrorChange()) {
            cout << "[StVO] Small optimization solution variance" << endl;
            break;
        }
        // update previous values
        //更新以前的值
        err_prev = err;
    }

    DT_cov = H.inverse();  //DT_cov = Matrix6d::Identity();
    err_   = err;
}

void StereoFrameHandler::gaussNewtonOptimizationRobust(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters)
{

    Matrix4d DT_;
    Matrix6d H;
    Vector6d g, DT_inc;
    double err, err_prev = 999999999.9;
    bool solution_is_good = true;
    DT_ = DT;

    // plot initial solution
    int iters;
    for( iters = 0; iters < max_iters; iters++)
    {
        // estimate hessian and gradient (select)
        optimizeFunctionsRobust( DT, H, g, err );
        // if the difference is very small stop
        if( ( fabs(err-err_prev) < Config::minErrorChange() ) || ( err < Config::minError()) )// || err > err_prev )
            break;
        // update step
        ColPivHouseholderQR<Matrix6d> solver(H);
        DT_inc = solver.solve(g);
        if( solver.logAbsDeterminant() < 0.0 || solver.info() != Success )
        {
            solution_is_good = false;
            break;
        }
        DT  << DT * inverse_se3( expmap_se3(DT_inc) );
        // if the parameter change is small stop (TODO: change with two parameters, one for R and another one for t)
        if( DT_inc.norm() < Config::minErrorChange() )
            break;
        // update previous values
        err_prev = err;
    }

    if( solution_is_good )
    {
        DT_cov = H.inverse();  //DT_cov = Matrix6d::Identity();
        err_   = err;
    }
    else
    {
        DT = DT_;
        err_ = -1.0;
        DT_cov = Matrix6d::Identity();
    }

}

void StereoFrameHandler::levenbergMarquardtOptimization(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters)
{
    Matrix6d H;
    Vector6d g, DT_inc;
    double err, err_prev = 999999999.9;

    double lambda   = 0.000000001;
    double lambda_k = 4.0;

    // form the first iteration
    optimizeFunctionsRobust( DT, H, g, err );
    //optimizeFunctions( DT, H, g, err );

    // initial guess of lambda
    double Hmax = 0.0;
    for( int i = 0; i < 6; i++)
    {
        if( H(i,i) > Hmax || H(i,i) < -Hmax )
            Hmax = fabs( H(i,i) );
    }
    lambda *= Hmax;

    // solve the first iteration
    for(int i = 0; i < 6; i++)
        H(i,i) += lambda;// * H(i,i) ;
    ColPivHouseholderQR<Matrix6d> solver(H);
    DT_inc = solver.solve(g);
    DT  << DT * inverse_se3( expmap_se3(DT_inc) );
    err_prev = err;

    // start Levenberg-Marquardt minimization
    //plotStereoFrameProjerr(DT,0);
    for( int iters = 1; iters < max_iters; iters++)
    {
        // estimate hessian and gradient (select)
        optimizeFunctionsRobust( DT, H, g, err );
        //optimizeFunctions( DT, H, g, err );
        // if the difference is very small stop
        if( ( fabs(err-err_prev) < Config::minErrorChange() ) || ( err < Config::minError()) )
            break;
        // add lambda to hessian
        for(int i = 0; i < 6; i++)
            H(i,i) += lambda;// * H(i,i) ;
        // update step
        ColPivHouseholderQR<Matrix6d> solver(H);
        DT_inc = solver.solve(g);
        // update lambda
        if( err > err_prev )
            lambda /= lambda_k;
        else
        {
            lambda *= lambda_k;
            DT  << DT * inverse_se3( expmap_se3(DT_inc) );
        }
        // plot each iteration
        //plotStereoFrameProjerr(DT,iters+1);
        // if the parameter change is small stop
        if( DT_inc.head(3).norm() < Config::minErrorChange() && DT_inc.tail(3).norm() < Config::minErrorChange())
            break;
        // update previous values
        err_prev = err;
    }

    DT_cov = H.inverse();  //DT_cov = Matrix6d::Identity();
    err_   = err;
}

//优化函数体
//输入：DT：上一帧到当前帧的转移矩阵 H：hessian矩阵 g：梯度向量 e ：残差
void StereoFrameHandler::optimizeFunctions(Matrix4d DT, Matrix6d &H, Vector6d &g, double &e )
{

    // define hessians, gradients, and residuals
    Matrix6d H_l, H_p; //线特征hessian矩阵、点特征hessian矩阵
    Vector6d g_l, g_p;  //线特征梯度向量、点特征梯度向量
    double   e_l = 0.0, e_p = 0.0, S_l, S_p; //线特征残差
    H   = Matrix6d::Zero(); H_l = H; H_p = H;  //初始化值
    g   = Vector6d::Zero(); g_l = g; g_p = g;  //初始化值
    e   = 0.0;//初始化残差

    // point features
    //点特征
    int N_p = 0;

    //遍历匹配点，将所有内点的重投影误差叠加起来，同时构造雅可比矩阵
    for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++)
    {
        //如果该特征点是内点
        if( (*it)->inlier )
        {
            Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);//将该点特征投影到当前帧的坐标系下
            Vector2d pl_proj = cam->projection( P_ ); //将相机坐标转化为像素坐标
            // projection error
            //重投影误差
            Vector2d err_i    = pl_proj - (*it)->pl_obs; //计算误差
            double err_i_norm = err_i.norm(); //计算2范数
            // estimate variables for J, H, and g
            double gx   = P_(0); //相机坐标的x
            double gy   = P_(1); //相机坐标的y
            double gz   = P_(2); //相机坐标的z
            double gz2  = gz*gz;  //z*z
            double fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2);  //getFx()：返回相机的fx参数  fx/(z*z)
            double dx   = err_i(0); //误差的x
            double dy   = err_i(1);  //误差的y
            // jacobian
            //雅可比矩阵
            Vector6d J_aux;
            J_aux << + fgz2 * dx * gz,
                     + fgz2 * dy * gz,
                     - fgz2 * ( gx*dx + gy*dy ),
                     - fgz2 * ( gx*gy*dx + gy*gy*dy + gz*gz*dy ),
                     + fgz2 * ( gx*gx*dx + gz*gz*dx + gx*gy*dy ),
                     + fgz2 * ( gx*gz*dy - gy*gz*dx );
            J_aux = J_aux / std::max(Config::homogTh(),err_i_norm);
            // define the residue
            double s2 = (*it)->sigma2;
            double r = err_i_norm * sqrt(s2);
            // if employing robust cost function
            //如果应用了鲁棒代价函数
            double w  = 1.0;
            w = robustWeightCauchy(r) ;

            // if down-weighting far samples
            //double zdist   = P_(2) * 1.0 / ( cam->getB()*cam->getFx());
            //if( false ) w *= 1.0 / ( s2 + zdist * zdist );

            // update hessian, gradient, and error
            H_p += J_aux * J_aux.transpose() * w;
            g_p += J_aux * r * w;
            e_p += r * r * w;
            N_p++;
        }
    }

    // line segment features
    //线特征
    int N_l = 0;
    //遍历所有线匹配对，将所有内线特征的重投影误差叠加起来，并构造雅可比矩阵
    for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++)
    {
        //如果该线特征是内线特征
        if( (*it)->inlier )
        {
            Vector3d sP_ = DT.block(0,0,3,3) * (*it)->sP + DT.col(3).head(3);  //将起始端点投影到当前帧的相机坐标系中
            Vector2d spl_proj = cam->projection( sP_ ); //将相机坐标转化为像素坐标
            Vector3d eP_ = DT.block(0,0,3,3) * (*it)->eP + DT.col(3).head(3);//将结束端点投影到当前帧的相机坐标系中
            Vector2d epl_proj = cam->projection( eP_ );//将相机坐标转化为像素坐标
            Vector3d l_obs = (*it)->le_obs;//取出该线特征匹配的当前帧的线特征
            // projection error
            //重投影误差
            Vector2d err_i;
            err_i(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);//起始点到直线的距离
            err_i(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2); //结束点到直线的距离
            double err_i_norm = err_i.norm(); //求2范数
            // estimate variables for J, H, and g
            // -- start point
            double gx   = sP_(0); //起始点相机坐标的x
            double gy   = sP_(1);  //起始点相机坐标的y
            double gz   = sP_(2);  //起始点相机坐标的z
            double gz2  = gz*gz;  //z*z
            double fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2); //fx/(z*z)
            double ds   = err_i(0); //误差的x
            double de   = err_i(1);  //误差的y
            double lx   = l_obs(0);  //直线方程的x
            double ly   = l_obs(1);    //直线方程的y
            Vector6d Js_aux;
            Js_aux << + fgz2 * lx * gz,
                      + fgz2 * ly * gz,
                      - fgz2 * ( gx*lx + gy*ly ),
                      - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
                      + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
                      + fgz2 * ( gx*gz*ly - gy*gz*lx );
            // -- end point
            gx   = eP_(0);//结束点相机坐标的x
            gy   = eP_(1);//结束点相机坐标的y
            gz   = eP_(2);//结束点相机坐标的z
            gz2  = gz*gz; //z*z
            fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2); //fx/(z*z)
            Vector6d Je_aux, J_aux;
            Je_aux << + fgz2 * lx * gz,
                      + fgz2 * ly * gz,
                      - fgz2 * ( gx*lx + gy*ly ),
                      - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
                      + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
                      + fgz2 * ( gx*gz*ly - gy*gz*lx );
            // jacobian
            J_aux = ( Js_aux * ds + Je_aux * de ) / std::max(Config::homogTh(),err_i_norm);  //总的雅可比矩阵

            // define the residue
            //定义残差
            double s2 = (*it)->sigma2;
            double r = err_i_norm * sqrt(s2);

            // if employing robust cost function
            double w  = 1.0;
            w = robustWeightCauchy(r) ;

            // estimating overlap between line segments
            //估计线段之间的重叠
            bool has_overlap = true;
            double overlap = prev_frame->lineSegmentOverlap( (*it)->spl, (*it)->epl, spl_proj, epl_proj ); //计算重叠部分的长度
            if( has_overlap )
                w *= overlap;

            // if down-weighting far samples
            /*double zdist = max( sP_(2), eP_(2) ) / ( cam->getB()*cam->getFx());
            if( false )
                w *= 1.0 / ( s2 + zdist * zdist );*/

            // update hessian, gradient, and error
            H_l += J_aux * J_aux.transpose() * w;
            g_l += J_aux * r * w;
            e_l += r * r * w;
            N_l++;
        }

    }

    // sum H, g and err from both points and lines
    H = H_p + H_l;
    g = g_p + g_l;
    e = e_p + e_l;

    // normalize error
    e /= (N_l+N_p);

}

void StereoFrameHandler::optimizeFunctionsRobust(Matrix4d DT, Matrix6d &H, Vector6d &g, double &e )
{

    // define hessians, gradients, and residuals
    Matrix6d H_l, H_p;
    Vector6d g_l, g_p;
    double   e_l = 0.0, e_p = 0.0, S_l, S_p;
    H   = Matrix6d::Zero(); H_l = H; H_p = H;
    g   = Vector6d::Zero(); g_l = g; g_p = g;
    e   = 0.0;

    vector<double> res_p, res_l;

    // point features pre-weight computation
    //点特征预权重计算
    for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++)
    {
        if( (*it)->inlier )
        {
            Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);
            Vector2d pl_proj = cam->projection( P_ );
            // projection error
            //投影误差
            Vector2d err_i    = pl_proj - (*it)->pl_obs;
            res_p.push_back( err_i.norm());
        }
    }

    // line segment features pre-weight computation
    //线特征预权重计算
    for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++)
    {
        if( (*it)->inlier )
        {
            Vector3d sP_ = DT.block(0,0,3,3) * (*it)->sP + DT.col(3).head(3);
            Vector2d spl_proj = cam->projection( sP_ );
            Vector3d eP_ = DT.block(0,0,3,3) * (*it)->eP + DT.col(3).head(3);
            Vector2d epl_proj = cam->projection( eP_ );
            Vector3d l_obs = (*it)->le_obs;
            // projection error
            //线特征投影误差
            Vector2d err_i;
            err_i(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
            err_i(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
            res_l.push_back( err_i.norm() );
        }

    }

    // estimate scale of the residuals
    double s_p = 1.0, s_l = 1.0;
    double th_min = 0.0001;
    double th_max = sqrt(7.815);

    if( false )
    {

        res_p.insert( res_p.end(), res_l.begin(), res_l.end() );
        s_p = vector_stdv_mad( res_p );

        //cout << s_p << "\t";
        //if( res_p.size() < 4*Config::minFeatures() )
            //s_p = 1.0;
        //cout << s_p << endl;

        if( s_p < th_min )
            s_p = th_min;
        if( s_p > th_max )
            s_p = th_max;

        s_l = s_p;

    }
    else
    {

        s_p = vector_stdv_mad( res_p ); //得到投影误差的标准偏差，反映投影误差的离散程度
        s_l = vector_stdv_mad( res_l );

        if( s_p < th_min )
            s_p = th_min;
        if( s_p > th_max )
            s_p = th_max;

        if( s_l < th_min )
            s_l = th_min;
        if( s_l > th_max )
            s_l = th_max;

    }

    bool useWeight=false;//是否使用不确定权重标志位，true：使用，false:不使用
    // point features
    //点特征
    int N_p = 0;
    for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++)
    {
        if( (*it)->inlier )
        {
            Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);
            Vector2d pl_proj = cam->projection( P_ );
            // projection error
            Vector2d err_i    = pl_proj - (*it)->pl_obs;
            double err_i_norm = err_i.norm();
            // estimate variables for J, H, and g
            double gx   = P_(0);//x
            double gy   = P_(1);//y
            double gz   = P_(2);//z
            double gz2  = gz*gz;//z^2
            double fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2);//f/z^2
            double dx   = err_i(0);//dx
            double dy   = err_i(1);//dy
            // jacobian
            Vector6d J_aux;
            J_aux << + fgz2 * dx * gz,
                     + fgz2 * dy * gz,
                     - fgz2 * ( gx*dx + gy*dy ),
                     - fgz2 * ( gx*gy*dx + gy*gy*dy + gz*gz*dy ),
                     + fgz2 * ( gx*gx*dx + gz*gz*dx + gx*gy*dy ),
                     + fgz2 * ( gx*gz*dy - gy*gz*dx );
            J_aux = J_aux / std::max(Config::homogTh(),err_i_norm);
            // define the residue
            double s2 = (*it)->sigma2;
            double r = err_i_norm ;
            // if employing robust cost function
            double w  = 1.0;
            double x = r / s_p;
            w = robustWeightCauchy(x) ;

            // if using uncertainty weights
            //如果使用不确定权重
            //----------------------------------------------------
            
            if( useWeight )
            {
                Matrix2d covp;
                Matrix3d covP_ = (*it)->covP_an;
                MatrixXd J_Pp(2,3), J_pr(1,2);
                // uncertainty of the projection
                J_Pp  << gz, 0.f, -gx, 0.f, gz, -gy;
                J_Pp  << J_Pp * DT.block(0,0,3,3);
                covp  << J_Pp * covP_ * J_Pp.transpose();
                covp  << covp / std::max(Config::homogTh(),gz2*gz2);    // Covariance of the 3D projection 三维投影的协方差\hat{p} up to f2*b2*sigma2
                covp  = sqrt(s2) * cam->getB()* cam->getFx() * covp;
                covp(0,0) += s2;
                covp(1,1) += s2;
                // Point Uncertainty constants 点不确定性常数
                /*bsigmaP   = f * baseline * sigmaP;
                bsigmaP   = bsigmaP * bsigmaP;
                bsigmaP_inv   = 1.f / bsigmaP;
                sigmaP2       = sigmaP * sigmaP;
                sigmaP2_inv   = 1.f / sigmaP2;
                // Line Uncertainty constants 线段不确定性常数
                bsigmaL   = baseline * sigmaL;
                bsigmaL   = bsigmaL * bsigmaL;
                bsigmaL_inv   = 1.f / bsigmaL;*/
                // uncertainty of the residual
                J_pr << dx / r, dy / r;
                double cov_res = (J_pr * covp * J_pr.transpose())(0);
                cov_res = 1.0 / std::max(Config::homogTh(),cov_res);
                double zdist   = P_(2) * 1.0 / ( cam->getB()*cam->getFx());

                //zdist   = 1.0 / std::max(Config::homogTh(),zdist);
                /*cout.setf(ios::fixed,ios::floatfield); cout.precision(8);
                cout << endl << cov_res << " " << 1.0 / cov_res << " " << zdist << " " << 1.0 / zdist << " " << 1.0 / (zdist*40.0) << "\t"
                     << 1.0 / ( 1.0 +  cov_res * cov_res + zdist * zdist ) << " \t"
                     << 1.0 / ( cov_res * cov_res + zdist * zdist )
                     << endl;
                cout.setf(ios::fixed,ios::floatfield); cout.precision(3);*/
                //w *= 1.0 / ( cov_res * cov_res + zdist * zdist );
                w *= 1.0 / ( s2 + zdist * zdist );
            }


            //----------------------------------------------------

            // update hessian, gradient, and error
            H_p += J_aux * J_aux.transpose() * w;
            g_p += J_aux * r * w;
            e_p += r * r * w;
            N_p++;
        }
    }

    // line segment features
    int N_l = 0;
    for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++)
    {
        if( (*it)->inlier )
        {
            Vector3d sP_ = DT.block(0,0,3,3) * (*it)->sP + DT.col(3).head(3);
            Vector2d spl_proj = cam->projection( sP_ );
            Vector3d eP_ = DT.block(0,0,3,3) * (*it)->eP + DT.col(3).head(3);
            Vector2d epl_proj = cam->projection( eP_ );
            Vector3d l_obs = (*it)->le_obs;
            // projection error
            Vector2d err_i;
            err_i(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
            err_i(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
            double err_i_norm = err_i.norm();
            // estimate variables for J, H, and g
            // -- start point
            double gx   = sP_(0);
            double gy   = sP_(1);
            double gz   = sP_(2);
            double gz2  = gz*gz;
            double fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2);
            double ds   = err_i(0);
            double de   = err_i(1);
            double lx   = l_obs(0);
            double ly   = l_obs(1);
            Vector6d Js_aux;
            Js_aux << + fgz2 * lx * gz,
                      + fgz2 * ly * gz,
                      - fgz2 * ( gx*lx + gy*ly ),
                      - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
                      + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
                      + fgz2 * ( gx*gz*ly - gy*gz*lx );
            // -- end point
            gx   = eP_(0);
            gy   = eP_(1);
            gz   = eP_(2);
            gz2  = gz*gz;
            fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2);
            Vector6d Je_aux, J_aux;
            Je_aux << + fgz2 * lx * gz,
                      + fgz2 * ly * gz,
                      - fgz2 * ( gx*lx + gy*ly ),
                      - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
                      + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
                      + fgz2 * ( gx*gz*ly - gy*gz*lx );
            // jacobian
            J_aux = ( Js_aux * ds + Je_aux * de ) / std::max(Config::homogTh(),err_i_norm);
            // define the residue
            double s2 = (*it)->sigma2;
            double r = err_i_norm;
            // if employing robust cost function
            //如果采用鲁棒代价函数
            double w  = 1.0;
            double x = r / s_l;
            w = robustWeightCauchy(x) ;
            // estimating overlap between line segments
            //估计线段之间的重叠 
            bool has_overlap = true;
            double overlap = prev_frame->lineSegmentOverlap( (*it)->spl, (*it)->epl, spl_proj, epl_proj );
            if( has_overlap )
                w *= overlap;

            //----------------- DEBUG: 27/11/2017 ----------------------
            if( useWeight )
            {
                //double cov3d = (*it)->cov3d;
                //cov3d = 1.0;
                //w *= 1.0 / ( 1.0 +  cov3d * cov3d + zdist * zdist );
                double zdist = max( sP_(2), eP_(2) ) / ( cam->getB()*cam->getFx());
                w *= 1.0 / ( s2 + zdist * zdist );
            }
            //----------------------------------------------------------

            // update hessian, gradient, and error
            H_l += J_aux * J_aux.transpose() * w;
            g_l += J_aux * r * w;
            e_l += r * r * w;
            N_l++;
        }

    }

    // sum H, g and err from both points and lines
    H = H_p + H_l;
    g = g_p + g_l;
    e = e_p + e_l;
    
    //double lweight=0.5*((double)N_l/(double)(N_p+N_l));
    //cout<<"N_p="<<N_p<<"N_l="<<N_l<<endl;
    //cout<<"lweight="<<setprecision(8)<<lweight<<endl;
    //e = (1-lweight)*e_p + lweight*e_l;

    // normalize error
    e /= (N_l+N_p);

}

void StereoFrameHandler::resetOutliers() {

    for (auto pt : matched_pt)
        pt->inlier = true;
    for (auto ls : matched_ls)
        ls->inlier = true;

    n_inliers_pt = matched_pt.size();
    n_inliers_ls = matched_ls.size();
    n_inliers    = n_inliers_pt + n_inliers_ls;
}

void StereoFrameHandler::setAsOutliers() {

    for (auto pt : matched_pt)
        pt->inlier = false;
    for (auto ls : matched_ls)
        ls->inlier = false;

    n_inliers_pt = 0;
    n_inliers_ls = 0;
    n_inliers    = 0;
}

//去除异常点
//通过计算投影误差的绝对中位差来得到异常值误差的阈值，当投影误差大于该阈值时，判定该特征是异常特征，剔除
void StereoFrameHandler::removeOutliers(Matrix4d DT)
{

    //TODO: if not usig mad stdv, use just a fixed threshold (sqrt(7.815)) to filter outliers (with a single for loop...)
    //如果不使用mad stdv，则仅使用固定阈值（sqrt（7.815））来过滤异常值（使用单个for循环…）
    //stdv：基于样本估算标准偏差，标准偏差反映数值相对于平均值(mean) 的离散程度

    //如果提取了点特征
    if (Config::hasPoints()) {
        // point features
        vector<double> res_p;//投影误差容器
        res_p.reserve(matched_pt.size());//初始化容器大小为帧间点特征匹配的匹配对数量
        int iter = 0;
        //遍历每一对匹配点对
        for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
        {
            // projection error
            Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);//将该点特征投影到当前帧的坐标系下
            Vector2d pl_proj = cam->projection( P_ );//相机坐标->像素坐标
            res_p.push_back( ( pl_proj - (*it)->pl_obs ).norm() * sqrt((*it)->sigma2) );//norm()取2范数，像素误差的2范数再乘以一个系数
            //res_p.push_back( ( pl_proj - (*it)->pl_obs ).norm() );
        }
        // estimate robust parameters
        //估计鲁棒参数
        double p_stdv, p_mean, inlier_th_p;
        //对投影误差求绝对中位差（误差样本与中位数的绝对值，然后再对这些绝对值求中位数），用1.4826*绝对中位差得到p_stdv。
        //如果投影误差>2*p_stdv，说明该投影点是异常点，统计内点数量，如果内点数量大于0.2*总投影点数量，则计算内点误差的平均值p_mean
        //如果内点数量小于0.2*总投影点数量，即异常值太多，则计算全部投影点误差的平均值p_mean
        vector_mean_stdv_mad( res_p, p_mean, p_stdv );  
        inlier_th_p = Config::inlierK() * p_stdv; //判断是否是异常值的阈值 默认是1.2*stdv
        //inlier_th_p = sqrt(7.815);
        //cout << endl << p_mean << " " << p_stdv << "\t" << inlier_th_p << endl;
        // filter outliers
        iter = 0;
        //遍历每一对匹配对
        for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
        {
            //剔除掉判断为外点的特征点
            if( (*it)->inlier && fabs(res_p[iter]-p_mean) > inlier_th_p )
            {
                (*it)->inlier = false;
                n_inliers--;
                n_inliers_pt--;
            }
        }
    }

    //如果提取了线特征
    if (Config::hasLines()) {
        // line segment features
        vector<double> res_l; //线特征投影误差容器
        res_l.reserve(matched_ls.size());  
        int iter = 0;
        //遍历每一对线匹配对 计算线投影误差
        for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++, iter++)
        {
            // projection error
            Vector3d sP_ = DT.block(0,0,3,3) * (*it)->sP + DT.col(3).head(3);  //起始点投影到当前帧的相机坐标
            Vector3d eP_ = DT.block(0,0,3,3) * (*it)->eP + DT.col(3).head(3);  //结束点投影到当前帧的相机坐标
            Vector2d spl_proj = cam->projection( sP_ );//起始点相机坐标转化为像素坐标
            Vector2d epl_proj = cam->projection( eP_ );//结束点相机坐标转化为像素坐标
            Vector3d l_obs    = (*it)->le_obs;  //该线特征在当前帧下的匹配线
            Vector2d err_li; //误差
            err_li(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
            err_li(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
            res_l.push_back( err_li.norm() * sqrt( (*it)->sigma2 ) );  //线特征的投影误差
            //res_l.push_back( err_li.norm() );
        }

        // estimate robust parameters
        double l_stdv, l_mean, inlier_th_l;
        //对投影误差求绝对中位差（误差样本与中位数的绝对值，然后再对这些绝对值求中位数），用1.4826*绝对中位差得到p_stdv。
        //如果投影误差>2*p_stdv，说明该投影线是异常线，统计内线数量，如果内线数量大于0.2*总投影线数量，则计算内线误差的平均值p_mean
        //如果内线数量小于0.2*总投影线数量，即异常值太多，则计算全部投影线误差的平均值p_mean
        vector_mean_stdv_mad( res_l, l_mean, l_stdv );  
        inlier_th_l = Config::inlierK() * l_stdv; //判断是否是异常值的阈值
        //inlier_th_p = sqrt(7.815);
        //cout << endl << l_mean << " " << l_stdv << "\t" << inlier_th_l << endl << endl;

        // filter outliers
        iter = 0;
        //遍历每一对线匹配对
        for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++, iter++)
        {
            //剔除调判断为外线的线特征
            if( fabs(res_l[iter]-l_mean) > inlier_th_l  && (*it)->inlier )
            {
                (*it)->inlier = false;
                n_inliers--;
                n_inliers_ls--;
            }
        }
    }

    //如果点线特征数量！=点特征数量+线特征数量 则输出错误信息
    if (n_inliers != (n_inliers_pt + n_inliers_ls))
        throw runtime_error("[StVO; stereoFrameHandler] Assetion failed: n_inliers != (n_inliers_pt + n_inliers_ls)");
}

void StereoFrameHandler::prefilterOutliers(Matrix4d DT)
{

    vector<double> res_p, res_l, ove_l, rob_res_p, rob_res_l;

    // point features
    int iter = 0;
    for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
    {
        // projection error
        Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);
        Vector2d pl_proj = cam->projection( P_ );
        res_p.push_back( ( pl_proj - (*it)->pl_obs ).norm() * sqrt((*it)->sigma2) );
    }

    // line segment features
    iter = 0;
    for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++, iter++)
    {
        // projection error
        Vector3d sP_ = DT.block(0,0,3,3) * (*it)->sP + DT.col(3).head(3);
        Vector3d eP_ = DT.block(0,0,3,3) * (*it)->eP + DT.col(3).head(3);
        Vector2d spl_proj = cam->projection( sP_ );
        Vector2d epl_proj = cam->projection( eP_ );
        Vector3d l_obs    = (*it)->le_obs;
        Vector2d err_li;
        err_li(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
        err_li(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
        res_l.push_back( err_li.norm() * sqrt( (*it)->sigma2 ) );
    }

    // estimate mad standard deviation
    double p_mad = vector_stdv_mad( res_p );
    double l_mad = vector_stdv_mad( res_l );
    double p_mean = vector_mean_mad( res_p, p_mad, Config::inlierK() );
    double l_mean = vector_mean_mad( res_l, l_mad, Config::inlierK() );
    double inlier_th_p =  Config::inlierK() * p_mad;
    double inlier_th_l =  Config::inlierK() * l_mad;
    p_mean = 0.0;
    l_mean = 0.0;

    // filter outliers
    iter = 0;
    for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
    {
        if( res_p[iter] > inlier_th_p && (*it)->inlier )
        {
            (*it)->inlier = false;
            n_inliers--;
            n_inliers_pt--;
        }
    }
    iter = 0;
    for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++, iter++)
    {
        if( res_l[iter] > inlier_th_l  && (*it)->inlier )
        {
            (*it)->inlier = false;
            n_inliers--;
            n_inliers_ls--;
        }
    }

}

/*  slam functions  */
//判断是否需要关键帧
bool StereoFrameHandler::needNewKF()
{

    // if the previous KF was a KF, update the entropy_first_prevKF value
    if( prev_f_iskf )
    {
        if( curr_frame->DT_cov.determinant() != 0.0 )
        {
            entropy_first_prevKF = 3.0*(1.0+log(2.0*acos(-1))) + 0.5*log( curr_frame->DT_cov.determinant() );
            prev_f_iskf = false;
        }
        else
        {
            entropy_first_prevKF = -999999999.99;
            prev_f_iskf = false;
        }

    }

    // check geometric distances from previous KF
    Matrix4d DT = inverse_se3( curr_frame->Tfw ) * T_prevKF;
    Vector6d dX = logmap_se3( DT );
    double t = dX.head(3).norm();
    double r = dX.tail(3).norm() * 180.f / CV_PI;

    // check cumulated covariance from previous KF
    Matrix6d adjTprevkf = adjoint_se3( T_prevKF );
    Matrix6d covDTinv   = uncTinv_se3( curr_frame->DT, curr_frame->DT_cov );
    cov_prevKF_currF += adjTprevkf * covDTinv * adjTprevkf.transpose();
    double entropy_curr  = 3.0*(1.0+log(2.0*acos(-1))) + 0.5*log( cov_prevKF_currF.determinant() );
    double entropy_ratio = entropy_curr / entropy_first_prevKF;

    //cout << endl << curr_frame->DT     << endl << endl;
    //cout << endl << curr_frame->DT_cov << endl << endl;
    //cout << endl << cov_prevKF_currF   << endl << endl;

    // decide if a new KF is needed
    if( entropy_ratio < Config::minEntropyRatio() || std::isnan(entropy_ratio) || std::isinf(entropy_ratio) ||
        ( curr_frame->DT_cov == Matrix6d::Zero() && curr_frame->DT == Matrix4d::Identity() ) ||
        t > Config::maxKFTDist() || r > Config::maxKFRDist() || N_prevKF_currF > 10 )
    {
        cout << endl << "Entropy ratio: " << entropy_ratio   << "\t" << t << " " << r << " " << N_prevKF_currF << endl;
        return true;
    }
    else
    {
        cout << endl << "No new KF needed: " << entropy_ratio << "\t" << entropy_curr << " " << entropy_first_prevKF
             << " " << cov_prevKF_currF.determinant() << "\t" << t << " " << r << " " << N_prevKF_currF << endl << endl;
        N_prevKF_currF++;
        return false;
    }
}

//当前帧是关键帧
void StereoFrameHandler::currFrameIsKF()
{

    // restart point indices
    int idx_pt = 0;
    for( auto it = curr_frame->stereo_pt.begin(); it != curr_frame->stereo_pt.end(); it++)
    {
        (*it)->idx = idx_pt;
        idx_pt++;
    }

    // restart line indices
    int idx_ls = 0;
    for( auto it = curr_frame->stereo_ls.begin(); it != curr_frame->stereo_ls.end(); it++)
    {
        (*it)->idx = idx_ls;
        idx_ls++;
    }

    // update KF
    curr_frame->Tfw     = Matrix4d::Identity();
    curr_frame->Tfw_cov = Matrix6d::Identity();

    // update SLAM variables for KF decision
    T_prevKF = curr_frame->Tfw;
    cov_prevKF_currF = Matrix6d::Zero();
    prev_f_iskf = true;
    N_prevKF_currF = 0;

}

/* Debug functions */

void StereoFrameHandler::plotLeftPair() {

    cout << "Prev frame: " << prev_frame->stereo_pt.size() << endl;
    cout << "Curr frame: " << curr_frame->stereo_pt.size() << endl;
    cout << "Matched: " << matched_pt.size() << endl;
    cout << "Inliers: " << n_inliers_pt << endl;

    Mat im1, im2;

    if (prev_frame->img_l.channels() == 1) {
        im1 = prev_frame->img_l;
    } else if (prev_frame->img_l.channels() == 3 || prev_frame->img_l.channels() == 4)
        cvtColor(prev_frame->img_l, im1, COLOR_BGR2GRAY);
    else
        throw runtime_error("[plotLeftPair] Unsupported number of (prev) image channels");
    im1.convertTo(im1, CV_8UC1);
    cvtColor(im1, im1, COLOR_GRAY2BGR);

    if (curr_frame->img_l.channels() == 1) {
        im2 = curr_frame->img_l;
    } else if (curr_frame->img_l.channels() == 3 || curr_frame->img_l.channels() == 4)
        cvtColor(curr_frame->img_l, im2, COLOR_BGR2GRAY);
    else
        throw runtime_error("[plotLeftPair] Unsupported number of (curr) image channels");
    im2.convertTo(im2, CV_8UC1);
    cvtColor(im2, im2, COLOR_GRAY2BGR);

    Scalar color;

    //plot stereo features
    color = Scalar(0, 255, 0);
    for (auto pt : prev_frame->stereo_pt)
        cv::circle(im1, cv::Point(pt->pl(0), pt->pl(1)), 3, color, -1);

    color = Scalar(0, 0, 255);
    for (auto pt : curr_frame->stereo_pt)
        cv::circle(im2, cv::Point(pt->pl(0), pt->pl(1)), 3, color, -1);

    //plot matched features
    random_device rnd_dev;
    mt19937 rnd(rnd_dev());
    uniform_int_distribution<int> color_dist(0, 255);

    for (auto pt : matched_pt) {
        color = Scalar(color_dist(rnd), color_dist(rnd), color_dist(rnd));
        cv::circle(im1, cv::Point(pt->pl(0), pt->pl(1)), 3, color, -1);
        cv::circle(im2, cv::Point(pt->pl_obs(0), pt->pl_obs(1)), 3, color, -1);
    }

    //putsidebyside
    Size sz1 = im1.size();
    Size sz2 = im2.size();

    Mat im(sz1.height, sz1.width+sz2.width, CV_8UC3);

    Mat left(im, Rect(0, 0, sz1.width, sz1.height));
    im1.copyTo(left);

    Mat right(im, Rect(sz1.width, 0, sz2.width, sz2.height));
    im2.copyTo(right);

    imshow("LeftPair", im);
}

void StereoFrameHandler::plotStereoFrameProjerr( Matrix4d DT, int iter )
{

    // create new image to modify it
    Mat img_l_aux_p, img_l_aux_l;
    curr_frame->img_l.copyTo( img_l_aux_p );

    if( img_l_aux_p.channels() == 3 )
        cvtColor(img_l_aux_p,img_l_aux_p,CV_BGR2GRAY);

    cvtColor(img_l_aux_p,img_l_aux_p,CV_GRAY2BGR);
    img_l_aux_p.copyTo( img_l_aux_l );

    // Variables
    Point2f         p,q,r,s;
    double          thick = 1.5;
    int             k = 0, radius  = 3;

    // plot point features
    for( auto pt_it = matched_pt.begin(); pt_it != matched_pt.end(); pt_it++)
    {
        if( (*pt_it)->inlier )
        {
            Vector3d P_ = DT.block(0,0,3,3) * (*pt_it)->P + DT.col(3).head(3);
            Vector2d pl_proj = cam->projection( P_ );
            Vector2d pl_obs  = (*pt_it)->pl_obs;

            p = cv::Point( int(pl_proj(0)), int(pl_proj(1)) );
            circle( img_l_aux_p, p, radius, Scalar(255,0,0), thick);

            q = cv::Point( int(pl_obs(0)), int(pl_obs(1)) );
            circle( img_l_aux_p, p, radius, Scalar(0,0,255), thick);

            line( img_l_aux_p, p, q, Scalar(0,255,0), thick);

        }
    }

    // plot line segment features
    for( auto ls_it = matched_ls.begin(); ls_it != matched_ls.end(); ls_it++)
    {
        if( (*ls_it)->inlier )
        {
            Vector3d sP_ = DT.block(0,0,3,3) * (*ls_it)->sP + DT.col(3).head(3);
            Vector2d spl_proj = cam->projection( sP_ );
            Vector3d eP_ = DT.block(0,0,3,3) * (*ls_it)->eP + DT.col(3).head(3);
            Vector2d epl_proj = cam->projection( eP_ );

            Vector2d spl_obs  = (*ls_it)->spl_obs;
            Vector2d epl_obs  = (*ls_it)->epl_obs;

            p = cv::Point( int(spl_proj(0)), int(spl_proj(1)) );
            q = cv::Point( int(epl_proj(0)), int(epl_proj(1)) );

            r = cv::Point( int(spl_obs(0)), int(spl_obs(1)) );
            s = cv::Point( int(epl_obs(0)), int(epl_obs(1)) );

            line( img_l_aux_l, p, q, Scalar(255,0,0), thick);
            line( img_l_aux_l, r, s, Scalar(0,0,255), thick);

            line( img_l_aux_l, p, r, Scalar(0,255,0), thick);
            line( img_l_aux_l, q, s, Scalar(0,255,0), thick);

            double overlap = prev_frame->lineSegmentOverlap( spl_obs, epl_obs, spl_proj, epl_proj );
            Vector2d mpl_obs = 0.5*(spl_obs+epl_obs);
            mpl_obs(0) += 4;
            mpl_obs(1) += 4;
            putText(img_l_aux_l,to_string(overlap),cv::Point(int(mpl_obs(0)),int(mpl_obs(1))),
                    FONT_HERSHEY_PLAIN, 0.5, Scalar(0,255,255) );


        }
    }

    //string pwin_name = "Iter: " + to_string(iter);
    string pwin_name = "Points";
    imshow(pwin_name,img_l_aux_p);
    string lwin_name = "Lines";
    imshow(lwin_name,img_l_aux_l);
    waitKey(0);


}

void StereoFrameHandler::optimizePoseDebug()
{

    // definitions
    Matrix6d DT_cov;
    Matrix4d DT, DT_;
    Vector6d DT_cov_eig;
    double   err;

    // set init pose
    DT     = prev_frame->DT;
    DT_cov = prev_frame->DT_cov;

    DT = Matrix4d::Identity();
    DT_cov = Matrix6d::Zero();

    // solver
    if( n_inliers > Config::minFeatures() )
    {
        // optimize
        DT_ = DT;
        gaussNewtonOptimizationRobustDebug(DT_,DT_cov,err,Config::maxIters());
        // remove outliers (implement some logic based on the covariance's eigenvalues and optim error)
        if( is_finite(DT_) && err != -1.0 )
        {
            removeOutliers(DT_);
            // refine without outliers
            if( n_inliers > Config::minFeatures() )
                gaussNewtonOptimizationRobustDebug(DT,DT_cov,err,Config::maxItersRef());
            else
            {
                DT     = Matrix4d::Identity();
                DT_cov = Matrix6d::Zero();
            }

        }
        else
        {
            DT     = Matrix4d::Identity();
            DT_cov = Matrix6d::Zero();
        }
    }
    else
    {
        DT     = Matrix4d::Identity();
        DT_cov = Matrix6d::Zero();
    }

    // set estimated pose
    if( is_finite(DT) && err != -1.0 )
    {
        curr_frame->DT     = inverse_se3( DT );
        curr_frame->Tfw    = prev_frame->Tfw * curr_frame->DT;
        curr_frame->DT_cov = DT_cov;
        SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
        curr_frame->DT_cov_eig = eigensolver.eigenvalues();
        curr_frame->Tfw_cov = unccomp_se3( prev_frame->Tfw, prev_frame->Tfw_cov, DT_cov );
        curr_frame->err_norm   = err;
    }
    else
    {
        curr_frame->DT     = Matrix4d::Identity();
        curr_frame->DT_cov = Matrix6d::Identity();
        curr_frame->err_norm   = -1.0;
        curr_frame->Tfw    = prev_frame->Tfw;
        curr_frame->Tfw_cov= prev_frame->Tfw_cov;
        SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
        curr_frame->DT_cov_eig = eigensolver.eigenvalues();
    }

    // ************************************************************************************************ //
    // 1. IDENTIFY ANOTHER FUCKING CRITERIA FOR DETECTING JUMPS                                         //
    // 2. INTRODUCE VARIABLE TO INDICATE THAT IN THAT KF ESTIMATION YOU'VE JUMPED SEVERAL FRAMES -> PGO //
    // ************************************************************************************************ //
    // cout << endl << prev_frame->DT_cov_eig.transpose() << endl;                                      //
    // cout << endl << prev_frame->DT                     << endl;                                      //
    /*SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
    Vector6d max_eig = eigensolver.eigenvalues();
    SelfAdjointEigenSolver<Matrix3d> eigensolverT( DT_cov.block(0,0,3,3) );
    SelfAdjointEigenSolver<Matrix3d> eigensolverR( DT_cov.block(3,3,3,3) );
    Vector3d max_eig_t = eigensolverT.eigenvalues();
    Vector3d max_eig_r = eigensolverR.eigenvalues();
    Vector6d shur = logmap_se3(DT);
    cout.setf(ios::fixed,ios::floatfield); cout.precision(8);
    cout << endl << max_eig << endl;
    //cout << endl << max_eig(3) / max_eig(0) << endl;
    //cout << endl << max_eig(4) / max_eig(1) << endl;
    //cout << endl << max_eig(5) / max_eig(2) << endl << endl;
    cout << endl << shur.head(3).norm() << endl;
    cout << endl << shur.tail(3).norm() << endl;
    cout << endl << max_eig_t * shur.head(3).norm()  << endl;
    cout << endl << max_eig_r * shur.tail(3).norm() << endl << endl;
    cout.setf(ios::fixed,ios::floatfield); cout.precision(3);*/
    //getchar();
    // ************************************************************************************************ //

    // set estimated pose
    if( is_finite(DT) )
    {
        curr_frame->DT     = inverse_se3( DT );
        curr_frame->Tfw    = prev_frame->Tfw * curr_frame->DT;
        curr_frame->DT_cov = DT_cov;
        SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
        curr_frame->DT_cov_eig = eigensolver.eigenvalues();
        curr_frame->Tfw_cov = unccomp_se3( prev_frame->Tfw, prev_frame->Tfw_cov, DT_cov );
        curr_frame->err_norm   = err;
    }
    else
    {
        curr_frame->DT     = Matrix4d::Identity();
        curr_frame->Tfw    = prev_frame->Tfw;
        curr_frame->Tfw_cov= prev_frame->Tfw_cov;
        curr_frame->DT_cov = DT_cov;
        SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
        curr_frame->DT_cov_eig = eigensolver.eigenvalues();
        curr_frame->err_norm   = -1.0;
    }

}

void StereoFrameHandler::gaussNewtonOptimizationRobustDebug(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters)
{
    Matrix6d H;
    Vector6d g, DT_inc;
    double err, err_prev = 999999999.9;
    bool solution_is_good = true;
    Matrix4d DT_;

    DT_ = DT;

    // plot initial solution
    plotStereoFrameProjerr(DT,0);
    int iters;
    for( iters = 0; iters < max_iters; iters++)
    {
        // estimate hessian and gradient (select)
        optimizeFunctionsRobust( DT, H, g, err );
        // if the difference is very small stop
        if( ( fabs(err-err_prev) < Config::minErrorChange() ) || ( err < Config::minError()) )// || err > err_prev )
            break;
        // update step
        //LDLT<Matrix6d> solver(H);
        ColPivHouseholderQR<Matrix6d> solver(H);
        DT_inc = solver.solve(g);
        if( solver.logAbsDeterminant() < 10.0 || solver.info() != Success )
        {
            solution_is_good = false;
            cout << endl << "Cuidao shur" << endl;
            getchar();
            break;
        }
        DT  << DT * inverse_se3( expmap_se3(DT_inc) );
        // plot each iteration
        plotStereoFrameProjerr(DT,iters+1);
        // if the parameter change is small stop (TODO: change with two parameters, one for R and another one for t)
        if( DT_inc.norm() < Config::minErrorChange() )
            break;
        // update previous values
        err_prev = err;
    }

    if( solution_is_good )
    {
        DT_cov = H.inverse();  //DT_cov = Matrix6d::Identity();
        err_   = err;
    }
    else
    {
        DT = DT_;
        err_ = -1.0;
        DT_cov = Matrix6d::Identity();
    }

}

}
