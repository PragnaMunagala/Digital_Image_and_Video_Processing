package com.example.munag.eee508androidopencv;

import java.util.Arrays;
import android.graphics.Color;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.widget.TextView;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import static com.example.munag.eee508androidopencv.R.*;

public class MainActivity extends AppCompatActivity implements View.OnTouchListener, CameraBridgeViewBase.CvCameraViewListener2{
    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat mRgba;
    public static final int HISTOGRAM = 3, CANNY = 1, SOBEL = 2, ZOOM = 4, INIT_MODE = 0;
    private MenuItem itemHistogram, itemCanny, itemSobel, itemZoom;
    public static int menuSelected = INIT_MODE;
    double x = -1;
    double y = -1;
    private Mat tempMat, initMat;
    private MatOfInt noOfChannels[], histSize;
    private int histSizeNum = 25;
    private MatOfFloat range;
    TextView touch_coordinates;
    TextView touch_color;
    private Scalar colorRGB[], colorHue[], colorWhite;
    private Point p1, p2;
    private float imgBuffer[];
    private Scalar mBlobColorRgba;
    private Scalar mBlobColorHsv;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(MainActivity.this);
                } break;
                default: {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        //Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        touch_coordinates = (TextView) findViewById(R.id.touch_coordinates);
        touch_color = (TextView) findViewById(R.id.touch_color);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_tutorial_activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public boolean onTouch(View view, MotionEvent motionEvent) {
        int cols = mRgba.cols();
        int rows = mRgba.rows();
        double yLow = (double)mOpenCvCameraView.getHeight() * 0.2401961;
        double yHigh = (double)mOpenCvCameraView.getHeight() * 0.7696078;
        double xScale = (double)cols / (double)mOpenCvCameraView.getWidth();
        double yScale = (double)rows / (yHigh - yLow);
        x = motionEvent.getX();
        y = motionEvent.getY();
        y = y -yLow;
        x = x * xScale;
        y = y * yScale;

        if((x < 0) || (y < 0) || (x > cols) || (y > rows))
            return false;

        touch_coordinates.setText("X: "+ Double.valueOf(x) + " Y: " + Double.valueOf(y));

        Rect touchedRect = new Rect();
        touchedRect.x = (int)x;
        touchedRect.y = (int)y;
        touchedRect.width = 8;
        touchedRect.height = 8;

        Mat touchedRegionRgba = mRgba.submat(touchedRect);
        Mat touchedRegionHsv = new Mat();
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);
        mBlobColorHsv = Core.sumElems(touchedRegionHsv);
        int pointCount = touchedRect.width * touchedRect.height;
        for(int i = 0; i < mBlobColorHsv.val.length; i++)
            mBlobColorHsv.val[i] /= pointCount;

        mBlobColorRgba = convertScalarHsv2Rgba(mBlobColorHsv);
        touch_color.setText("Color: #" + String.format("%02X", (int)mBlobColorRgba.val[0]) +
                String.format("%02X", (int)mBlobColorRgba.val[1]) +
                        String.format("%02X", (int)mBlobColorRgba.val[2]));
        touch_color.setTextColor(Color.rgb((int)mBlobColorRgba.val[0],
                (int)mBlobColorRgba.val[1], (int)mBlobColorRgba.val[2]));
        touch_coordinates.setTextColor(Color.rgb((int)mBlobColorRgba.val[0],
                (int)mBlobColorRgba.val[1], (int)mBlobColorRgba.val[2]));

        return false;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        itemCanny = menu.add("Canny Filtering");
        itemSobel = menu.add("Sobel Filtering");
        itemHistogram  = menu.add("Histogram");
        itemZoom  = menu.add("Zoom In");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item == itemHistogram)
            menuSelected = HISTOGRAM;
        else if (item == itemCanny)
            menuSelected = CANNY;
        else if (item == itemSobel)
            menuSelected = SOBEL;
        else if (item == itemZoom)
            menuSelected = ZOOM;
        return true;
    }

    private void HistogramPlot(Size sizeRgba) {
        Mat hist = new Mat();
        int tness = (int) (sizeRgba.width / (histSizeNum + 10) / 5);
        if(tness > 5) tness = 5;
        int offset = (int) ((sizeRgba.width - (5*histSizeNum + 4*10)*tness)/2);
        int c = 0;
        while(c<3) {
            Imgproc.calcHist(Arrays.asList(mRgba), noOfChannels[c], initMat, hist, histSize, range);
            Core.normalize(hist, hist, sizeRgba.height/2, 0, Core.NORM_INF);
            hist.get(0, 0, imgBuffer);
            int h = 0;
            while(h<histSizeNum) {
                p1.x = p2.x = offset + (c * (histSizeNum + 10) + h) * tness;
                p1.y = sizeRgba.height-1;
                p2.y = p1.y - 2 - (int)imgBuffer[h];
                Imgproc.line(mRgba, p1, p2, colorRGB[c], tness);
                h++;
            }
            c++;
        }
        Imgproc.cvtColor(mRgba, tempMat, Imgproc.COLOR_RGB2HSV_FULL);
        Imgproc.calcHist(Arrays.asList(tempMat), noOfChannels[2], initMat, hist, histSize, range);
        Core.normalize(hist, hist, sizeRgba.height/2, 0, Core.NORM_INF);
        hist.get(0, 0, imgBuffer);
        for(int h=0; h<histSizeNum; h++) {
            p1.x = p2.x = offset + (3 * (histSizeNum + 10) + h) * tness;
            p1.y = sizeRgba.height-1;
            p2.y = p1.y - 2 - (int)imgBuffer[h];
            Imgproc.line(mRgba, p1, p2, colorWhite, tness);
        }
        Imgproc.calcHist(Arrays.asList(tempMat), noOfChannels[0], initMat, hist, histSize, range);
        Core.normalize(hist, hist, sizeRgba.height/2, 0, Core.NORM_INF);
        hist.get(0, 0, imgBuffer);
        for(int h=0; h<histSizeNum; h++) {
            p1.x = p2.x = offset + (4 * (histSizeNum + 10) + h) * tness;
            p1.y = sizeRgba.height-1;
            p2.y = p1.y - 2 - (int)imgBuffer[h];
            Imgproc.line(mRgba, p1, p2, colorHue[h], tness);
        }
    }

    private void CannyPlot(Mat tempRgba) {
        Imgproc.Canny(tempRgba, tempMat, 70, 80);
        Imgproc.cvtColor(tempMat, tempRgba, Imgproc.COLOR_GRAY2BGRA, 4);
    }

    private void SobelPlot(Mat grayInnerWindow, Mat tempRgba) {
        Imgproc.Sobel(grayInnerWindow, tempMat, CvType.CV_8U, 1, 1);
        Core.convertScaleAbs(tempMat, tempMat, 10, 0);
        Imgproc.cvtColor(tempMat, tempRgba, Imgproc.COLOR_GRAY2BGRA, 4);
    }

    private void ZoomPlot(int rows, int cols) {
        Mat tempImage = mRgba.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);
        Mat tempWindow = mRgba.submat(rows / 2 - 9 * rows / 100, rows / 2 + 9 * rows / 100, cols / 2 - 9 * cols / 100, cols / 2 + 9 * cols / 100);
        Imgproc.resize(tempWindow, tempImage, tempImage.size());
        Size box = tempWindow.size();
        Imgproc.rectangle(tempWindow, new Point(1, 1), new Point(box.width - 2, box.height - 2), new Scalar(255, 0, 0, 255), 1);
        tempImage.release();
        tempWindow.release();
    }

    private Scalar convertScalarHsv2Rgba(Scalar hsvColor){
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);
         return new Scalar(pointMatRgba.get(0, 0));
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mBlobColorRgba = new Scalar(255);
        mBlobColorHsv = new Scalar(255);
        tempMat = new Mat();
        noOfChannels = new MatOfInt[] { new MatOfInt(0), new MatOfInt(1), new MatOfInt(2) };
        imgBuffer = new float[histSizeNum];
        histSize = new MatOfInt(histSizeNum);
        range = new MatOfFloat(0f, 256f);
        initMat  = new Mat();
        colorRGB = new Scalar[] { new Scalar(200, 0, 0, 255), new Scalar(0, 200, 0, 255), new Scalar(0, 0, 200, 255) };
        colorHue = new Scalar[] {
                new Scalar(255, 0, 0, 255), new Scalar(255, 60, 0, 255),  new Scalar(255, 120, 0, 255), new Scalar(255, 180, 0, 255), new Scalar(255, 240, 0, 255),
                new Scalar(215, 213, 0, 255), new Scalar(150, 255, 0, 255), new Scalar(85, 255, 0, 255),  new Scalar(20, 255, 0, 255),  new Scalar(0, 255, 30, 255),
                new Scalar(0, 255, 85, 255),  new Scalar(0, 255, 150, 255), new Scalar(0, 255, 215, 255), new Scalar(0, 234, 255, 255), new Scalar(0, 170, 255, 255),
                new Scalar(0, 120, 255, 255), new Scalar(0, 60, 255, 255),  new Scalar(0, 0, 255, 255),   new Scalar(64, 0, 255, 255),  new Scalar(120, 0, 255, 255),
                new Scalar(180, 0, 255, 255), new Scalar(255, 0, 255, 255), new Scalar(255, 0, 215, 255), new Scalar(255, 0, 85, 255),  new Scalar(255, 0, 0, 255)
        };
        colorWhite = Scalar.all(255);
        p1 = new Point();
        p2 = new Point();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Mat tempRgba;
        Size sizeRgba = mRgba.size();
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;
        int left = cols/20;
        int top = rows/20;
        int width = cols*9/10;
        int height = rows*9/10;

        switch (MainActivity.menuSelected) {
            case MainActivity.HISTOGRAM:
                HistogramPlot(sizeRgba);
                break;
            case MainActivity.CANNY:
                tempRgba = mRgba.submat(top, top + height, left, left + width);
                CannyPlot(tempRgba);
                tempRgba.release();
                break;
            case MainActivity.SOBEL:
                Mat gray = inputFrame.gray();
                Mat grayInnerWindow = gray.submat(top, top + height, left, left + width);
                tempRgba = mRgba.submat(top, top + height, left, left + width);
                SobelPlot(grayInnerWindow, tempRgba);
                grayInnerWindow.release();
                tempRgba.release();
                break;
            case MainActivity.ZOOM:
               ZoomPlot(rows, cols);
                break;
        }
        return mRgba;
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
    }

    @Override
    public void onPause(){
        super.onPause();
        if(mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume(){
        super.onResume();
        if(!OpenCVLoader.initDebug())
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        else
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
    }

    @Override
    public void onDestroy(){
        super.onDestroy();
        if(mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }
}
