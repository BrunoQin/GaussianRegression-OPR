package com.tongji.gp.helper;

import Jama.Matrix;
import com.csvreader.CsvWriter;
import ucar.ma2.Array;
import ucar.nc2.NetcdfFile;
import ucar.nc2.Variable;

import java.nio.charset.Charset;
import java.util.Arrays;

public class FileHelper {

    private static final String ROOTPATH = "/BIGDATA1/iocas_mmu_2/";
    private static final String READPATH = ROOTPATH + "GFDL-CM2p1/";
    private static final String WRITEPATH = ROOTPATH + "/PPSO/record/workspace/";
    private static final String PARAMETER = "sst";

    public static Matrix prepareFile(){

        for(int i = 10; i < 90; i++){

            String fileName;
            if((5 * i + 1) < 100){
                fileName = READPATH + "output/00" + (5 * i + 1) + "0101.ocean_month.nc";
            } else {
                fileName = READPATH + "output/0" + (5 * i + 1) + "0101.ocean_month.nc";
            }


            try{

                NetcdfFile ncfile = NetcdfFile.open(fileName);
                Variable sst = ncfile.findVariable(PARAMETER);

                for(int j = 0; j < 5; j++){

                    String dataFilePath = WRITEPATH + "data/";

                    for(int k = 0; k < 12; k++){
                        Array part = sst.read((j * 12 + k) + ":" + (j * 12 + k) + ":1, 0:199:1, 0:359:1");
                        float[][] content = (float[][])part.reduce().copyToNDJavaArray();
                        writeFile(content, dataFilePath + (5 * i + j + 1) + "." + (k + 1) + ".csv");
                    }

                }

            } catch (Exception e){
                e.printStackTrace();
            }


        }

        return null;
    }

    public static void writeFile(float[][] content, String fileName){

        try{

            CsvWriter csvWriter = new CsvWriter(fileName, ',', Charset.forName("UTF-8"));

            for(int i = 0; i < content.length; i++){
                String s = Arrays.toString(content[i]);
                s = s.substring(1, s.length() - 1);
                String[] s_array = s.split(", ");
                csvWriter.writeRecord(s_array);
            }

            csvWriter.close();
            System.out.println("--------" + fileName + "文件已经写入--------");

        } catch (Exception e){
            e.printStackTrace();
        }

    }

}
