package hr.foi.tiwo;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class OdInitializerCpu {
	
	public native void sendImageForProcessing(long image, int width, int height);
	private static String defaultPath = "/home/ivan/git/ObjectDetectionCPUGPU/";
	
	static{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.loadLibrary("DetectCpu");
		//System.loadLibrary("DetectGpu");
	}
	

	public void detect(String path) {

	    BufferedImage img = null;
	    try {
	        img = ImageIO.read(new File(path));
	    } catch (IOException e) { }
	    
	    Mat matImage = ObjDec.img2Mat(img);
	    long addr = matImage.getNativeObjAddr();
	    ObjDec.sendImageForProcessing(addr, img.getWidth(), img.getHeight());
	    
	    // result 
	    BufferedImage i = mat2Img(matImage);
	    // find out if object is detected...
	    //dispatch event:
	    System.out.println("Done with detection...");
	    
	    /*
		File outputfile = new File("java-output.png");
		
		try {
			ImageIO.write(i, "png", outputfile);
		} catch (IOException e) { }
		System.out.println("Done!");
		*/
	    
	}
	
	public static Mat img2Mat(BufferedImage image)  
    {  
		  byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();  
		  Mat mat = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);  
		  mat.put(0, 0, data);  
		  
		  return mat;  
     }
	
	public static BufferedImage mat2Img(Mat mat) {  
	     BufferedImage image = new BufferedImage(mat.width(), mat.height(), BufferedImage.TYPE_3BYTE_BGR);  
	     WritableRaster raster = image.getRaster();  
	     DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();  
	     byte[] data = dataBuffer.getData();  
	     mat.get(0, 0, data);  
	     return image;  
	  }
	


}
