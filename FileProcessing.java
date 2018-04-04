package source;

import java.io.FileNotFoundException;
import java.io.PrintWriter;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
//import moa.classifiers.meta.AccuracyUpdatedEnsemble;
import moa.core.InstanceExample;
import moa.evaluation.WindowAUCImbalancedPerformanceEvaluator;
import moa.streams.ArffFileStream;

public class FileProcessing {
	protected static int saPos = 0, saNeg=0, nSaPos = 0, nSaNeg=0; //number of true values
	protected static int tpDeprived=0, fpDeprived=0, tpFavored=0, fpFavored=0; 
	protected static int tnDeprived=0, fnDeprived=0, tnFavored=0, fnFavored=0;
	
	protected static Classifier learner = new NaiveBayes();//1.	
//	protected static Classifier learner = new HoeffdingTree();//2.
//	protected static Classifier learner = new kNN();//3.
//	protected static Classifier learner = new OzaBag();//4.
//	protected static Classifier learner = new AccuracyUpdatedEnsemble();//5.
	protected static int saIndex=0;
	protected static String saVal = "Female";
	protected static int desiredClass = 1;
	protected static int notDesiredClass = 0;
	
	protected static String saName = "sex";
	protected static int windowSize=0;
	protected static InstanceExample[] windowList = new InstanceExample[windowSize];	
	
	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub
		windowSize=1000;
		saName = "sex";
		saVal = "Female";	
//		String infile = "new_dataset_for_han_removefnlwgt_stream.arff";
		String infile = "small_dataset_remove_fnlwgt.arff";
		String outfile = "small_dataset_remove_fnlwgt_SA.arff";
		remove_att(infile, outfile, 8);
		
//		String infile="small_dataset_remove_fnlwgt_SAClass.arff";
//		String out="out/small_dataset_statistics.csv";
//		Statistics(infile,out);
		
//		String out="out/small_originalData(basedOnSAClass)_AUE.csv";
//		Data(infile, out);
	}
	public static void Data(String filename, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(filename, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();
		learner.setModelContext(fs.getHeader());		
		learner.prepareForUse();
		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();
				
		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, D_DeprivedPos, D_FavoredPos, D_DeprivedNeg, D_FavoredNeg, DisClassifier, C_DeprivedPos, C_FavoredPos, C_DeprivedNeg, C_FavoredNeg, Acc, prArea_Original, rocArea_Original, F1");
		
		int numberSamples = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			numberSamples++;									
								
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);			
			learner.trainOnInstance(trainInst);
							
			if (numberSamples>=windowSize)	{			
				if (numberSamples==windowSize||(numberSamples-windowSize)%10==0){					
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					int[] rankedLabels = evaluator.getAucEstimator().getsortedLabels();																		
					Curve curve = new Curve(rankedLabels);
					double prArea = curve.prArea();
							
					//for calculation of discrimination score
					String[] labels = evaluator.getAucEstimator().getSAVal();
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					double discData = Disc_Data();
					
					int saPos_C=tpDeprived+fpDeprived;
					int saNeg_C=tnDeprived+fnDeprived;
					int nSaPos_C=tpFavored+fpFavored;
					int nSaNeg_C=tnFavored+fnFavored;
					writer.println(numberSamples+","+discData+","+saPos+","+nSaPos+","+saNeg+","+nSaNeg+","+discClassifier+","+saPos_C+","+nSaPos_C+","+saNeg_C+","+nSaNeg_C+","+acc+","+prArea+","+rocArea+","+f1);
				}
			}
		}
		writer.close();
		System.out.println("Done Data");	
	}
	public static double DiscriminationScore(String[] labels, double[] predictions, int[] trueLabels){
		tpDeprived=0;tnDeprived=0;fnDeprived=0;fpDeprived=0;
		tpFavored=0;tnFavored=0;fnFavored=0;fpFavored=0;
		for (int i=0; i<windowSize; i++){
			if (labels[i].equals(saVal)){ //Deprived
				if (predictions[i]==1.0){//correctly predicted
					if (trueLabels[i]==desiredClass)//positive
						tpDeprived++;
					else
						tnDeprived++;
				}else{//incorrectly predicted
					if (trueLabels[i]==desiredClass)//positive => predict to => negative
						fnDeprived++;
					else
						fpDeprived++;
				}
			}else{//Favored
				if (predictions[i]==1.0){//correctly predicted
					if (trueLabels[i]==desiredClass)//positive
						tpFavored++;
					else
						tnFavored++;
				}else{//incorrectly predicted
					if (trueLabels[i]==desiredClass)//positive => predict to => negative
						fnFavored++;
					else
						fpFavored++;
				}
			}
		}
		saPos=tpDeprived+fnDeprived;
		saNeg=tnDeprived+fpDeprived;
		nSaPos=tpFavored+fnFavored;		
		nSaNeg=tnFavored+fpFavored;
		if ((saPos+saNeg)==0)
			return 100*(double)(tpFavored+fpFavored)/(double)(nSaPos+nSaNeg);
		else{
			if ((nSaPos+nSaNeg)==0)
				return -(double)(tpDeprived+fpDeprived)/(double)(saPos+saNeg);
			else 
				return 100*((double)(tpFavored+fpFavored)/(double)(nSaPos+nSaNeg)
					-(double)(tpDeprived+fpDeprived)/(double)(saPos+saNeg));
		}
	}	
	
	public static double Disc_Data(){//DiscriminationScore should be calculated first
		saPos=tpDeprived+fnDeprived;
		saNeg=tnDeprived+fpDeprived;
		nSaPos=tpFavored+fnFavored;		
		nSaNeg=tnFavored+fpFavored;		
		
		if ((saPos+saNeg)==0)
			return 100*(double)nSaPos/(double)(nSaPos+nSaNeg);
		else{
			if ((nSaPos+nSaNeg)==0)
				return -(double)saPos/(double)(saPos+saNeg);
			else 
				return 100*((double)(nSaPos)/(double)(nSaPos+nSaNeg)
						-(double)(saPos)/(double)(saPos+saNeg));
		}
			
	}
	public static void Statistics(String input, String output) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(input, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();
		int dp=0,dn=0,fp=0,fn=0;		
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			String[] splits=trainInst.toString().split(",");
			if (splits[saIndex].equals(saVal)){ //deprived
				if (Integer.parseInt(splits[splits.length-1])==desiredClass){//positive
					dp++;
				}else
					dn++;
			}else{//favored
				if (Integer.parseInt(splits[splits.length-1])==desiredClass){//positive
					fp++;
				}else
					fn++;
			}
		}
		PrintWriter w=new PrintWriter(output);
		w.println("dp,dn,fp,fn");
		w.println(dp+","+dn+","+fp+","+fn);
		w.close();
	}
	public static void remove_att(String infile, String outfile, int num) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);	
//		fs.prepareForUse();
		PrintWriter writer = new PrintWriter(outfile);
		writer.write(fs.getHeader().toString());		
		while (fs.hasMoreInstances()){
			Instance inst = fs.nextInstance().instance;
			String[] splits = inst.toString().split(",");
			String tmp="";
			for (int i=0; i<num; i++){
				tmp+=splits[i]+",";
			}
			for (int i=num+1; i<splits.length; i++){
				tmp+=splits[i]+",";
			}
			tmp=tmp.substring(0, tmp.length()-1);
			writer.println(tmp);
		}
		writer.close();
	}

}
