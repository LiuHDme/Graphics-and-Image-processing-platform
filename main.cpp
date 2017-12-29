#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <Glui2/glui2.h>
#include <GLUT/glut.h>
#include <vector>
#include <math.h>
#include <iostream>

using namespace std;
using namespace cv;

/*** 全局变量 ***/

// 窗口大小
const int WindowWidth = 1200;
const int WindowHeight = 600;

// 全局 Glui2 句柄
Glui2* GluiHandle = NULL;

// 2D 或 3D 图形的选择按钮
g2RadioGroup* DimensionContreller = NULL;

// 渲染方式
// 0 - 点, 1 - 线, 2 - 平面
g2DropDown* StyleController = NULL;

// 距离 (0) and 斜度 (1, 2)
g2Spinner* PositionSliders[3];
g2Label* PositionLabels[3];     // 标题

// 平移、旋转和缩放
g2Spinner* Transformation2DSliders[4];
g2Label* TranslationLabels[2];

// 距离和斜度
float Distance = 0, Pitch = 0, Yaw = 0;
// 平移量、旋转量和缩放量
float XTranslation = 0, YTranslation = 0, Rotate = 0, Scale = 1;

// 颜色的滑尺和标题
g2Slider* ColorSliders[3];
g2Label* ColorLabels[3];

// 裁剪控制按钮
g2RadioGroup* ClippedContreller = NULL;

// 2D 图形参数控件
g2DropDown* GraphController = NULL;

// 图像，依次为：原图、灰度图、直方图均衡化后的图、傅里叶变换后的图（无法直接查看）、反傅里叶变换后的图
Mat pic, pic_gray, pic_equalized, dft_container, inverse_dft;

// 裁剪窗口的位置
float xwl = -5, xwr = 5, ywb = -5, ywt = 5;

// CS 算法中的区域码
typedef int OutCode;

const int INSIDE = 0; // 0000
const int LEFT = 1;   // 0001
const int RIGHT = 2;  // 0010
const int BOTTOM = 4; // 0100
const int TOP = 8;    // 1000

/*** 全局函数 ***/

// 计算区域码
OutCode ComputeOutCode(double x, double y) {
    OutCode code;
    code = INSIDE;          // initialised as being inside of clip window
    
    if (x < xwl)           // to the left of clip window
        code |= LEFT;
    else if (x > xwr)      // to the right of clip window
        code |= RIGHT;
    if (y < ywb)           // below the clip window
        code |= BOTTOM;
    else if (y > ywt)      // above the clip window
        code |= TOP;
    
    return code;
}

// CS 算法
void CSLineClip(double x0, double y0, double x1, double y1) {
    // 计算两条线段的区域码
    OutCode outcode0 = ComputeOutCode(x0, y0);
    OutCode outcode1 = ComputeOutCode(x1, y1);
    bool accept = false;
    
    while (true) {
        if (!(outcode0 | outcode1)) {
            //相或为0，接受并且退出循环
            accept = true;
            break;
        } else if (outcode0 & outcode1) {
            // 相与为1，拒绝且退出循环
            break;
        } else {
            double x, y;
            
            //找出在界外的点
            OutCode outcodeOut = outcode0 ? outcode0 : outcode1;
            
            // 找出和边界相交的点
            // 使用点斜式 y = y0 + slope * (x - x0), x = x0 + (1 / slope) * (y - y0)
            if (outcodeOut & TOP) {           // point is above the clip rectangle
                x = x0 + (x1 - x0) * (ywt - y0) / (y1 - y0);
                y = ywt;
            } else if (outcodeOut & BOTTOM) { // point is below the clip rectangle
                x = x0 + (x1 - x0) * (ywb - y0) / (y1 - y0);
                y = ywb;
            } else if (outcodeOut & RIGHT) {  // point is to the right of clip rectangle
                y = y0 + (y1 - y0) * (xwr - x0) / (x1 - x0);
                x = xwr;
            } else if (outcodeOut & LEFT) {   // point is to the left of clip rectangle
                y = y0 + (y1 - y0) * (xwl - x0) / (x1 - x0);
                x = xwl;
            }
            
            if (outcodeOut == outcode0) {
                x0 = x;
                y0 = y;
                outcode0 = ComputeOutCode(x0, y0);
            } else {
                x1 = x;
                y1 = y;
                outcode1 = ComputeOutCode(x1, y1);
            }
        }
    }
    if (accept) {
        // 绘制 2D 图形
        glBegin(GL_LINE_LOOP);
            glVertex2f(x0, y0);
            glVertex2f(x1, y1);
        glEnd();
    }
}

// Liang-Barsky 裁剪算法
int LBLineClipTest(float p, float q, float &umax, float &umin) {
    float r = 0.0;
    if (p < 0.0) {
        r = q / p;
        if (r > umin) {
            return 0;
        }
        else if (r > umax) {
            umax = r;
        }
    }
    else if (p > 0.0) {
        r = q / p;
        if (r < umax) {
            return 0;
        }
        else if (r < umin) {
            umin = r;
        }
    }
    else if (q < 0.0) {
        return 0;
    }
    return 1;
}

// 使用 LB 裁剪算法
void LBLineClip(float x1, float y1, float x2, float y2) {
    float umax, umin, deltax, deltay;
    deltax = x2 - x1;
    deltay = y2 - y1;
    umax = 0.0;
    umin = 1.0;
    if (LBLineClipTest(-deltax, x1 - xwl, umax, umin))
        if (LBLineClipTest(deltax, xwr - x1, umax, umin))
            if (LBLineClipTest(-deltay, y1 - ywb, umax, umin))
                if (LBLineClipTest(deltay, ywt - y1, umax, umin)) {
                    float xx1 = x1, yy1 = y1; // 避免因 x1，y1 已变化而导致 x2，y2 不正确
                    x1 = x1 + umax * deltax;
                    y1 = y1 + umax * deltay;
                    x2 = xx1 + umin * deltax;
                    y2 = yy1 + umin * deltay;
                    
                    // 绘制 2D 图形
                    glBegin(GL_LINES);
                        glVertex2f(x1, y1);
                        glVertex2f(x2, y2);
                    glEnd();
                }
}

// 渲染
void Render() {

    // 定义裁减平面方程系数
    GLdouble eqn1[4] = {1.0, 0.0, 0.0, 0.0};
    GLdouble eqn2[4] = {1.0, 1.0, 0.0, 0.0};
    
    // 清除后台缓冲
    glClearColor(0.92f, 0.94f, 0.97f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    
    // 准备渲染
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0f, float(WindowWidth) / float(WindowHeight), 1.0f, 1000.0f);
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_DEPTH);
    
    // 设置点的大小
    glPointSize(3.0f);
    
    // 设置平移量和旋转量
    XTranslation = Transformation2DSliders[0]->GetFloat();
    YTranslation = Transformation2DSliders[1]->GetFloat();
    Rotate = Transformation2DSliders[2]->GetFloat();
    Scale = Transformation2DSliders[3]->GetFloat();
    
    // 更新距离和倾斜度 (3D)
    Distance = PositionSliders[0]->GetFloat();
    Pitch = PositionSliders[1]->GetFloat();
    Yaw = PositionSliders[2]->GetFloat();
    
    glPushMatrix();

        // 设置颜色
        glColor3f(ColorSliders[0]->GetProgress(), ColorSliders[1]->GetProgress(), ColorSliders[2]->GetProgress());

        // 根据用户的选择绘制图形
        if (DimensionContreller->GetSelectionIndex() == 1) {
            
            // 设置裁减平面
            if (ClippedContreller->GetSelectionIndex() == 1) {
                glClipPlane(GL_CLIP_PLANE0, eqn1);
                glEnable(GL_CLIP_PLANE0);
            } else if (ClippedContreller->GetSelectionIndex() == 2) {
                //设置裁减平面
                glClipPlane(GL_CLIP_PLANE0, eqn2);
                glEnable(GL_CLIP_PLANE0);
            } else {
                glDisable(GL_CLIP_PLANE0);
            }
            
            // 改变位置
            gluLookAt(-10.0f * Distance, 4 * Distance, -5.0f * Distance, 0, 0, 0, 0, 1, 0);
            glRotatef(Pitch, 1, 0, 0);
            glRotatef(Yaw, 0, 1, 0);
            
            // 改变渲染方式
            if(StyleController->GetSelectionIndex() == 0)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            else if(StyleController->GetSelectionIndex() == 1)
                glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
            else
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

            // 绘制 3D 图形
            glutSolidTeapot(3);

        } else {

            // 重新设置位置，不产生 3D 效果
            gluLookAt(0, 0, 100, 0, 0, 0, 0, 1, 0);
            
            // 平移、旋转和缩放
            glTranslatef(XTranslation, YTranslation, 0);
            glRotatef(Rotate, 0, 0, 1);
            glScalef(Scale, Scale, 1);
            
            float x1 = -10, y1 = 0, x2 = 10, y2 = 0, x3 = 0, y3 = 10;
            
            xwl = -5 - XTranslation;
            xwr = 5 - XTranslation;
            ywb = -5 - YTranslation;
            ywt = 5 - YTranslation;
            
            if (ClippedContreller->GetSelectionIndex() == 1) {
                // CS 裁剪
                CSLineClip(x1, y1, x2, y2);
                CSLineClip(x2, y2, x3, y3);
                CSLineClip(x1, y1, x3, y3);
            } else if (ClippedContreller->GetSelectionIndex() == 2) {
                // LB 裁剪
                LBLineClip(x1, y1, x2, y2);
                LBLineClip(x2, y2, x3, y3);
                LBLineClip(x1, y1, x3, y3);
            } else {
                // 绘制 2D 图形
                glBegin(GL_LINE_LOOP);
                    glVertex2f(x1, y1);
                    glVertex2f(x2, y2);
                    glVertex2f(x3, y3);
                glEnd();
            }
            
        }
    
    glPopMatrix();
    
    // 更新颜色选择滑块
    for (int i = 0; i < 3; i++) {
        char Buffer[256];
        sprintf(Buffer, "Color %c: %.2f%%", (i == 0 ? 'R' : (i == 1 ? 'G' : 'B')), ColorSliders[i]->GetProgress() * 100.0f);
        ColorLabels[i]->SetText(Buffer);
    }
    
    GluiHandle->Render();

    // 强制渲染，交换前后台缓冲区
    glFlush();
    glutSwapBuffers();
    
	// 重新渲染
	glutPostRedisplay();
}

// 重新缩放视口
void Reshape(int NewWidth, int NewHeight) {
	glViewport(0, 0, NewWidth, NewHeight);
}

// 退出
void Quit(g2Controller* Caller) {
    exit(0);
}

// 实施 Cohen－Sutherland 线段裁剪算法
void Clipping1(g2Controller* Caller) {}

/*** 弹窗的回调函数 ***/
void DialogOpen(g2Controller* Caller) {
    g2Dialog Dialog(g2DialogType_Open, "Open File...");
    Dialog.Show();
    
    // 得到结果
    char* String;
    int Result = (int)Dialog.GetInput(&String);
    printf("User's open-dialog result: %d (message: \"%s\")\n", Result, String);
    if (Result == 0) {
        pic = imread(String);
        namedWindow("pic");
        imshow("pic", pic);
    }
    delete[] String;
}

void DialogSave(g2Controller* Caller) {
    g2Dialog Dialog(g2DialogType_Save, "Save File...", "jpg");
    Dialog.Show();
    
    // 得到结果
//    char* String;
//    int Result = (int) Dialog.GetInput(&String);
//    const char* d = "/U";
//    if (Result == 0) {
//        Mat result;
//        inverse_dft.convertTo(result, CV_8UC3, 255);
//        char* s = strstr(String, d);
//        imwrite(s, result);
//    }
//    delete[] String;
}

// 转化为灰度图像并显示直方图
void transform2gray(g2Controller* Caller) {
    cvtColor(pic, pic_gray, COLOR_BGR2GRAY);
    imshow("pic", pic_gray);
    int bins = 256;
    int hist_size[] = {bins};
    float range[] = {0, 256};
    const float* ranges[] = {range};
    MatND hist;
    int channels[] = {0};
    calcHist(&pic_gray, 1, channels, Mat(), hist, 1, hist_size, ranges, true, false );
    double max_val;
    minMaxLoc(hist, 0, &max_val, 0, 0);
    int scale = 2;
    int hist_height = 256;
    Mat hist_img = Mat::zeros(hist_height,bins*scale, CV_8UC3);
    for(int i=0;i<bins;i++) {
        float bin_val = hist.at<float>(i);
        int intensity = cvRound(bin_val*hist_height/max_val);  //要绘制的高度
        rectangle(hist_img, cv::Point(i*scale,hist_height-1),
                  cv::Point((i+1)*scale - 1, hist_height - intensity),
                  CV_RGB(255,255,255));
    }
    namedWindow("hist");
    imshow("hist", hist_img);
}

// 显示均衡化后的图像和直方图
void EqualizeHist(g2Controller* Caller) {
    equalizeHist(pic_gray, pic_equalized);
    imshow("equalized", pic_equalized);
    int bins = 256;
    int hist_size[] = {bins};
    float range[] = {0, 256};
    const float* ranges[] = {range};
    MatND hist;
    int channels[] = {0};
    calcHist(&pic_equalized, 1, channels, Mat(), hist, 1, hist_size, ranges, true, false );
    double max_val;
    minMaxLoc(hist, 0, &max_val, 0, 0);
    int scale = 2;
    int hist_height = 256;
    Mat hist_img = Mat::zeros(hist_height,bins*scale, CV_8UC3);
    for(int i=0;i<bins;i++) {
        float bin_val = hist.at<float>(i);
        int intensity = cvRound(bin_val*hist_height/max_val);  // 要绘制的高度
        rectangle(hist_img, cv::Point(i*scale,hist_height-1),
                  cv::Point((i+1)*scale - 1, hist_height - intensity),
                  CV_RGB(255,255,255));
    }
    namedWindow("equalizedHist");
    imshow("equalizedHist", hist_img);
}

// 交换象限
void quadrantShift( Mat& src, Mat& dst) {
    
    int cx = src.cols/2;
    int cy = src.rows/2;
    
    cv::Rect q0 = cv::Rect(0, 0, cx, cy);   // Top-Left - Create a ROI per quadrant
    cv::Rect q1 = cv::Rect(cx, 0, cx, cy);  // Top-Right
    cv::Rect q2 = cv::Rect(0, cy, cx, cy);  // Bottom-Left
    cv::Rect q3 = cv::Rect(cx, cy, cx, cy); // Bottom-Right
    
    Mat temp;   // creating a temporary Mat object to protect the quadrant, in order to handle the situation where src = dst
    
    src(q0).copyTo(temp);   // preserve q0 section
    src(q3).copyTo(dst(q0));
    temp.copyTo(dst(q3));   // swap q0 and q3
    
    src(q1).copyTo(temp);
    src(q2).copyTo(dst(q1));
    temp.copyTo(dst(q2));
    
}

// 显示傅里叶转换或反转换后的图像
Mat visualDFT(Mat& dft_result) {
    Mat dst;
    // 创建一个包含两个矩阵的矩阵数组，用来保存幅度图和相位图
    Mat planes[2];
    // 把幅度图和相位图分别放入 planes 的两个矩阵中
    split(dft_result, planes);
    magnitude(planes[0], planes[1], dst);  // 此时的 dst 才可被显示
    
    // 用对数尺度替换线性尺度
    dst += Scalar::all(1);
    log(dst, dst);
    
    // 将float类型的矩阵转换到可显示图像范围
    // (float [0， 1])
    normalize(dst, dst,0,1,CV_MINMAX);
    
    // 交换象限
    quadrantShift(dst, dst);
    
    return dst;
}

// 离散傅里叶变换及巴特沃斯低通滤波
void dft(g2Controller* Caller) {
    Mat padded; // 将输入图像延扩到最佳的尺寸
    int m = getOptimalDFTSize(pic_gray.rows);
    int n = getOptimalDFTSize(pic_gray.cols);

    copyMakeBorder(pic_gray, padded, 0, m - pic_gray.rows, 0, n - pic_gray.cols, BORDER_CONSTANT, Scalar::all(0));  // 边缘填充 0
    
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};    // planes 包含两个矩阵，一个是 padded 的复制矩阵，一个是大小和 padded 一样的空矩阵
    merge(planes, 2, dft_container); // 将 planes 中的两个矩阵合并成 dft_container 矩阵
    
    dft(dft_container, dft_container);    // 对 dft_container 进行傅里叶变换
    
    // 把 dft_container 边像素变为偶数倍
    dft_container = dft_container(cv::Rect(0, 0, dft_container.cols & -2, dft_container.rows & -2));
    
    imshow("spectrum magnitude", visualDFT(dft_container));

    // 巴特沃斯低通滤波
    Mat butterWorth_filter(dft_container.size(), CV_32F);
    int r = butterWorth_filter.rows;
    int c = butterWorth_filter.cols;

    double D = 100;    // 通带半径
    double n_ButterWorth = 1.0/2;  // 巴特沃斯滤波器的系数
    
    // 根据公式进行计算
    // => H = 1 / (1 + (d / D) ^ (2 * n))
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            float d = sqrt((i - r/2) * (i - r/2) + (j - c/2) * (j - c/2));
            butterWorth_filter.at<float>(i, j) = 1.0 / (1 + pow(d / D, 2 * n_ButterWorth));
        }
    }
    
    // 合并
    Mat to_merge[] = {butterWorth_filter, butterWorth_filter};
    merge(to_merge, 2, butterWorth_filter);
    
    // 交换象限
    quadrantShift(butterWorth_filter, butterWorth_filter);

    // 进行巴特沃斯低通滤波
    mulSpectrums(dft_container, butterWorth_filter, dft_container, DFT_ROWS);
    
    imshow("After ButterWorth Filter", visualDFT(dft_container));
}

// 离散傅里叶反变换
void idft(g2Controller* Caller) {
    // 离散傅里叶反变换
    dft(dft_container, inverse_dft, DFT_INVERSE | DFT_REAL_OUTPUT);
    normalize(inverse_dft, inverse_dft, 0, 1, CV_MINMAX);
    imshow("idft", inverse_dft);
}

/*** 主函数和初始化函数 ***/

// 初始化 OpenGL 的 GLUT
void InitGLUT(int argc, char** argv) {
    // 初始化 glut
	glutInit(&argc, argv);
	
	// Double buffer w/ RGBA colors and z-depth turned on
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	
	// 居中窗口
	int SystemResWidth = glutGet(GLUT_SCREEN_WIDTH);
	int SystemResHeight = glutGet(GLUT_SCREEN_HEIGHT);
    glutInitWindowPosition(SystemResWidth / 2 - WindowWidth / 2, SystemResHeight / 2 - WindowHeight / 2);
    
	// 设置窗口大小
	glutInitWindowSize(WindowWidth, WindowHeight);
    
    // 生成窗口
    glutCreateWindow("课设");
    
    // 打开 alpha 通道
	glEnable(GL_ALPHA_TEST);
	glAlphaFunc(GL_GREATER, 0.01f);
	
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

// 初始化 Glui2 库
void InitGlui2() {
    // 生成 glui 的实例并注册必要的句柄
    GluiHandle = new Glui2("g2Default.cfg", NULL, Reshape);
    glutDisplayFunc(Render);
    
    /*** 选择裁剪算法 ***/
    const char* ClippedOptions[3];
    ClippedOptions[0] = "1. no clipping";
    ClippedOptions[1] = "2. Cohen-Sutherland";
    ClippedOptions[2] = "3. Liang-Barsky";
    
    ClippedContreller = GluiHandle->AddRadioGroup(20, 120, ClippedOptions, 3);
    
    /*** 选择 2D 图形或 3D 图形 ***/
    const char* Options[2];
    Options[0] = "2D";
    Options[1] = "3D";
    
    DimensionContreller = GluiHandle->AddRadioGroup(20, 40, Options, 2);
    
    /*** 颜色滑尺与标题 ***/
    for (int i = 0; i < 3; i++) {
        // 设置控件的位置和宽度
        ColorSliders[i] = GluiHandle->AddSlider(20, 520 + i * 25);
        ColorSliders[i]->SetWidth(170);
        
        ColorLabels[i] = GluiHandle->AddLabel(200, 520 + i * 25, "Color R/G/B");
        ColorLabels[i]->SetColor(0, 0, 0);
    }
    
    /*** 渲染方式 ***/
    const char* RenderingOptions[3];
    RenderingOptions[0] = "1. Line-rendering";
    RenderingOptions[1] = "2. Point-rendering";
    RenderingOptions[2] = "3. Surface-rendering";
    
    StyleController = GluiHandle->AddDropDown(20, 450, RenderingOptions, 3);
    StyleController->SetWidth(175);
    
    /*** 控制 2D 图形的变换***/
    for (int i = 0; i < 4; i++) {
        // 设置控件的位置和宽度
        Transformation2DSliders[i] = GluiHandle->AddSpinner(20, 200 + i * 25, g2SpinnerType_Float);
        Transformation2DSliders[i]->SetWidth(80);
        
        // 设置每次点击改变的大小
        Transformation2DSliders[i]->SetIncrement(1.0f);
        if (i == 3) {
            Transformation2DSliders[i]->SetIncrement(0.1f);
            Transformation2DSliders[i]->SetFloat(1.0f);
        }
        
        // 设置标题
        if (i == 0)
            TranslationLabels[i] = GluiHandle->AddLabel(115, 205 + i * 25, "XTranslation");
        else if (i == 1)
            TranslationLabels[i] = GluiHandle->AddLabel(115, 205 + i * 25, "YTranslation");
        else if (i == 2)
            TranslationLabels[i] = GluiHandle->AddLabel(115, 205 + i * 25, "Rotate");
        else
            TranslationLabels[i] = GluiHandle->AddLabel(115, 205 + i * 25, "Scale");
        TranslationLabels[i]->SetColor(0, 0, 0);
    }
    
    /*** 设置 2D 图形的参数 ***/
    const char* graph2DOptions[2];
    graph2DOptions[0] = "1. Triangle";
    graph2DOptions[1] = "2. Square";
    GraphController = GluiHandle->AddDropDown(20, 400, graph2DOptions, 2);
    GraphController->SetWidth(175);
    
    /*** 控制 3D 图形的变换 ***/
    for (int i = 0; i < 3; i++) {
        // 设置控件的位置和宽度
        PositionSliders[i] = GluiHandle->AddSpinner(20, 300 + i * 25, g2SpinnerType_Float);
        PositionSliders[i]->SetWidth(80);
        
        // 设置每次点击改变的大小
        if (i == 0) {
            PositionSliders[i]->SetFloat(1.0f);
            PositionSliders[i]->SetIncrement(0.05f);
        }
        else
            PositionSliders[i]->SetIncrement(5.0f);
        
        // 设置标题
        if (i == 0)
            PositionLabels[i] = GluiHandle->AddLabel(115, 305 + i * 25, "Distance");
        else if (i ==1)
            PositionLabels[i] = GluiHandle->AddLabel(115, 305 + i * 25, "Pitch");
        else
            PositionLabels[i] = GluiHandle->AddLabel(115, 305 + i * 25, "Yaw");
        PositionLabels[i]->SetColor(0, 0, 0);
    }
    
    /*** 设置 3D 图形的参数 ***/
    
    
    /*** 读取图像和保存图像 ***/
    GluiHandle->AddButton(WindowWidth - 100, 40, "1. Open File... ", DialogOpen);
    GluiHandle->AddButton(WindowWidth - 100, 60, "2. Save File... ", DialogSave);
    
    /*** 转化为灰度图像 ***/
    GluiHandle->AddButton(WindowWidth - 100, 100, "rgb2gray", transform2gray);
    
    /*** 直方图均衡化 ***/
    GluiHandle->AddButton(WindowWidth - 100, 140, "Equalize Hist", EqualizeHist);
    
    /*** 离散傅里叶变换 ***/
    GluiHandle->AddButton(WindowWidth - 100, 180, "DFT", dft);
    
    /*** 离散傅里叶反变换 ***/
    GluiHandle->AddButton(WindowWidth - 100, 220, "IDFT", idft);
    
    /*** 退出按钮 ***/
    GluiHandle->AddButton(WindowWidth - 100, WindowHeight - 50, "   Quit   ", Quit);
}

/*** 主函数 ***/
int main(int argc, char** argv) {
    // 初始化 OpenGL / GLUT
    InitGLUT(argc, argv);
    
    // 初始化 Glui2
    InitGlui2();
    
    // 开始主渲染循环
    glutMainLoop();
    
    return 0;
}
