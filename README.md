** You can download the sample images(coffee.jpg) to test the program.

** You need to install Opencv and set its the environmet in Visual studio 2022 first.

** You can check the results of image processin in AffineTrans.pdf.

** C++ code as below:
// Preset settings
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/core/utility.hpp>
#define pi  3.14159 // define the value of pi

using namespace std;
using namespace cv;

// Subroutines
int ShrinkOpt(int x_i, int y_i, int x_f, int y_f, Mat image, int imageSizeX, int imageSizeY)
{
    double PixWeightedSum = 0;
    double WeightedSum = 0;
    int AvgPix = 0;

    if (x_f >= imageSizeX)
    {
        x_f = imageSizeX-1;
    }

    if (y_f >= imageSizeY)
    {
        y_f = imageSizeY-1;
    }
    
    for (int i = y_i; i < y_f; i++)
    {
       uchar* data = image.ptr(i);
        for (int j = x_i; j < x_f; j++)
        {
            PixWeightedSum = PixWeightedSum + data[j] * 1;
            WeightedSum++;
        }

    }

    AvgPix = (PixWeightedSum / WeightedSum);
    
    return  AvgPix;
}

int Mediansort(int A[])
{
    for (int i = 0; i < 9; i++)
    {
        for (int j = i + 1; j < 9; j++)
        {
            if (A[i] > A[j])
            {
                int  temp = A[j];
                A[j] = A[i];
                A[i] = temp;
                temp = 0;
            }
        }
    }
    return A[4];
}

int main()
{
    // Preset settings of image processing
    Mat  image = imread("C:/opencv/Coffee.jpg", IMREAD_COLOR ); //read the image
    resize(image, image, {500, 380}, 0, 0, 1); // resize the read image
    Mat RGB[3]; // 
    split(image, RGB);
    Mat imageB = RGB[0];
    Mat imageBTrans = imageB.clone();
    Mat imageBTransOpt = imageB.clone();
    int col = imageB.size().width;
    int row = imageB.size().height;
    double  factor = 2; // if want to shrink half of image-->2, if want to enlage double of image--> 0.5
  
    int col2 = col / factor;
    int row2 = row / factor;
    
    uchar* data;
    uchar* dataTrans;
    uchar* dataTransOpt;

    // initialization of transformed image
    for (int i = 0; i < row; i++)
    {
        dataTrans = imageBTrans.ptr(i);
        dataTransOpt = imageBTransOpt.ptr(i);
  
        for (int j = 0; j < col; j++)
        {
            dataTrans[j] = 0;
            dataTransOpt[j] = 0;
          
        }
    }
    
    //declare and set the mask of local searching based on each processing pixel
    int LocalMask[9] = { -col - 1, -col, -col + 1 , -1, 0, 1, col - 1, col, col + 1 }; 
    int  Medianarray[9] = { 0 };

// enlarge image by factor 
    /*
   // 1. enlarge the image
    for (int i = 0; i < row; i++)
    {
        int y = factor * i;
        if (y > (row - 1))
        {
            y = (row - 1);
        }
        data = imageB.ptr(y);
        dataTrans = imageBTrans.ptr(i);
        for (int j = 0; j < col; j++)
        {
            if (i < (row / factor))
            {
                int x = factor * j;

                if (x > (col - 1))
                {
                    x = col - 1;
                    data[x] = 0;
                }
                dataTrans[j] = data[x];
            }
            else
            {
                dataTrans[j] = 0;
            }
        }
    }

    // 2. improve the image aliasing by using linear interpolation method 
    for (int i = 1; i < row - 1; i++)
    {
        dataTrans = imageBTrans.ptr(i);
        dataTransOpt = imageBTransOpt.ptr(i);
        for (int j = 1; j < col - 1; j++)
        {
            double row_u_inter = dataTrans[j - col - 1] + (dataTrans[j - col + 1] - dataTrans[j - col - 1]) / 2;
            double row_l_inter = dataTrans[j + col - 1] + (dataTrans[j + col + 1] - dataTrans[j + col - 1]) / 2;
            dataTransOpt[j] = row_u_inter + (row_l_inter - row_u_inter) / 2;
        }
    }
    */

// shrink image by factor under aliasing and optimazation
    /*
    for (int i = 0; i < row2;  i++)
    {
        int y = factor * i;
        int y_b = factor * (i + 1);
        data = imageB.ptr(y);
        dataTrans = imageBTrans.ptr(i);
        dataTransOpt = imageBTransOpt.ptr(i);
        for (int j = 0; j < col2;  j++)
        {
                int x = factor * j;
                int x_b = factor * (j + 1);

                dataTrans[j] = data[x];
               dataTransOpt[j] = ShrinkOpt(x, y, x_b, y_b, imageB, col, row);
        }
    }
    */


// rotation
    /*
    //1. rotate the image
    double theta =15;
  
    for (int i = 0; i < row; i++)
    {
        data = imageB.ptr(i);
        for (int j = 0; j< col; j++)
        {
            int x_theta = cos(theta/180*pi) *j+ sin(theta/180*pi) *i;
            int y_theta = -sin(theta/180*pi) *j + cos(theta/180*pi ) *i;
            if (x_theta >= col)
            {
                x_theta = col - 1;
            }
            if (x_theta <0)
            {
                x_theta = 0;
            }
            if (y_theta >= row)
            {
                y_theta = row - 1;
            }
            if (y_theta < 0)
            {
                y_theta = 0;
            }
            dataTrans = imageBTrans.ptr(y_theta);
            dataTrans[x_theta] = data[j];
        }

   }
   
   
// 2. improve the holes on image due to roration by using median filter
    /*
    for (int i = 1; i < row - 1; i++)
    {
        dataTrans = imageBTrans.ptr(i);
        dataTransOpt = imageBTransOpt.ptr(i);
        for (int j = 1; j < col - 1; j++)
        {
            for (int k = 0; k < 9; k++)
            {
                Medianarray[k] = dataTrans[j + LocalMask[k]];
            }

            dataTransOpt[j] = Mediansort(Medianarray);
        }
    }
    */


    // rotation by degree of 180
    for (int i = 0; i < row; i++)
    {
        data = imageB.ptr(i);
        dataTrans = imageBTrans.ptr(row-1-i);
        for (int j = 0; j < col; j++)
        {
            dataTrans[col-1-j] = data[j];
        }
    }

// set the image display
 Mat imageArray[] = {imageB, Mat(row, 5, CV_8UC1, Scalar(255)), imageBTrans, Mat(row, 5, CV_8UC1, Scalar(255)), imageBTransOpt}; // declare an image array to put images that will be showed.
 Mat comimage; // declare an image to put above images for showing.
 hconcat(imageArray, 5, comimage); // combine above images in horizontal.
 imshow("differences between before and after processing", comimage); // show the combined image.
 imwrite("differences between before and after processing.jpg", comimage);
 waitKey(0);
 
 return 0;

 }
 
