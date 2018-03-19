package source;

import java.io.FileNotFoundException;
import java.io.PrintWriter;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.core.InstanceExample;
import moa.evaluation.WindowAUCImbalancedPerformanceEvaluator;
import moa.streams.ArffFileStream;

public class Reweighing_StaticData {
	protected static int saPos = 0, saNeg=0, nSaPos = 0, nSaNeg=0; //number of true values
	protected static int tpDeprived=0, fpDeprived=0, tpFavored=0, fpFavored=0; 
	protected static int tnDeprived=0, fnDeprived=0, tnFavored=0, fnFavored=0;
	
	protected static double favPos=0;//weights 
	protected static double favNeg=0;
	protected static double savPos=0;
	protected static double savNeg=0;
	
	protected static int epsilon = 0; //percentage
	
	//different classifiers
	protected static Classifier learner = new NaiveBayes();//1.	
//	protected static Classifier learner = new HoeffdingTree();//2.
//	protected static Classifier learner = new kNN();//3.
//	protected static Classifier learner = new OzaBag();//4.
//	protected static Classifier learner = new AccuracyUpdatedEnsemble();//5.
//	protected static Classifier learner= new HoeffdingAdaptiveTree();
	//data definition
	protected static int saIndex=0;
	protected static String saVal = "Female";
	protected static int desiredClass = 1;
	protected static int notDesiredClass = 0;
	
	protected static String saName = "sex";
	protected static int windowSize=0;
	protected static int averageSize = 0;
	protected static InstanceExample[] windowList = new InstanceExample[windowSize];		
	
	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub
		windowSize=1000;
		averageSize=200;
		epsilon =0;
		saName = "sex";
		saVal = "Female";	
		
		String infile="new_dataset_for_han_removefnlwgt_stream.arff";
//		String outOriginal="out/original_"+windowSize+".csv";
//		Original(infile, outOriginal);
		//1.(YYY) Reset Learner + Continue Training + Apply New Weight after Reweighing
//		String outfile="out/"+averageSize+"_"+windowSize+"_YYY.csv";
//		Reweighing_ResetContTrainingDiffWeight(infile,outfile);
		
		//2.(YYN) Reset Learner + Continue Training + Use weight 1 after Reweighing
//		String outfile="out/"+averageSize+"_"+windowSize+"_YYN.csv";
//		Reweighing_ResetContTrainingSameWeight(infile,outfile);
		
		//3. (NYY) No Reset + Continue Training + Apply New Weight after Reweighing
//		String outfile="out/"+averageSize+"_"+windowSize+"_NYY.csv";
//		Reweighing_NoResetContTrainingDiffWeight(infile,outfile);
		
		//4. (NYN) No Reset + Continue Training + Use weight 1 after Reweighing
//		String outfile="out/"+averageSize+"_"+windowSize+"_NYN.csv";
//		Reweighing_NoResetContTrainingSameWeight(infile, outfile);
		
		//5. (YN_) Reset Learner + No Continue Training
//		String outfile="out/"+averageSize+"_"+windowSize+"_YN_.csv";
//		Reweighing_ResetNoContTraining(infile, outfile);
		
		//6. (NN_) No Reset + No Continue Training
		String outfile="out/"+averageSize+"_"+windowSize+"_NN_.csv";
		Reweighing_NoResetNoContTraining(infile, outfile);
		
		System.out.println("done main");
	}
	//0. Original Classifier
	public static void Original(String filename, String outfile) throws FileNotFoundException{
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
		writer.println("noSamples, DiscData_Original, DiscClassifier_Original, Acc_Original, P_Original, R_Original, F1_Original, rocArea_Original, prArea_Original");		
		
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
					double f1=2*precision*recall/(precision+recall);
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
							
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
									+precision+","+recall+","+f1+","+rocArea+","+prArea);
				}
			}
		}
		writer.close();
		System.out.println("Done Original");	
	}
	//1.(YYY) Reset Learner + Continue Training + Apply New Weight after Reweighing
	public static void Reweighing_ResetContTrainingDiffWeight(String filename, String outfile) throws FileNotFoundException{
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
		writer.println("noSamples, DiscData_YYY, DiscClassifier_YYY, Acc_YYY, P_YYY, R_YYY, F1_YYY, rocArea_YYY, prArea_YYY,favPos_YYY,favNeg_YYY,savPos_YYY,savNeg_YYY");
		windowList = new InstanceExample[windowSize];
		int numberSamples=0;
		double accumulatedDisc=0;double avgDisc = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			
			if (favPos!=0){						
				String[] splits=trainInst.toString().split(",");
				int cl=Integer.parseInt(splits[splits.length-1]);
				if (splits[saIndex].equals(saVal)){//Deprived					
					if (cl==desiredClass)//Positive class
						trainInst.setWeight(savPos);
					else
						trainInst.setWeight(savNeg);
				}else{
					if (cl==desiredClass)//Positive class
						trainInst.setWeight(favPos);
					else
						trainInst.setWeight(favNeg);
				}
			}
			learner.trainOnInstance(trainInst);			
			
			String[] labels;
			double[] predictions;
			int[] trueLabels;
			if (numberSamples>=windowSize){
				//for calculation of discrimination score
				labels = evaluator.getAucEstimator().getSAVal();
				predictions = evaluator.getAucEstimator().getPredictions();
				trueLabels = evaluator.getAucEstimator().getTrueLabel();
				double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
				double discData = Disc_Data();
				accumulatedDisc+=discClassifier;
				if (numberSamples==windowSize)
					avgDisc=discClassifier;
				else if ((numberSamples-windowSize)%averageSize==0)
					avgDisc=accumulatedDisc/averageSize;
				
				if ((numberSamples==windowSize)||((numberSamples-windowSize)%averageSize==0)){
					if (Math.abs(avgDisc)>epsilon){
						//reweighing
						int[] posWindow = evaluator.getAucEstimator().getPosWindowFromSortedScores();
						
						learner.resetLearning();
						ApplyReweighing(posWindow,labels, trueLabels);	
						//re-evaluate
						for (int i=0; i<windowSize; i++){
							InstanceExample instanceExample = windowList[i];
							Instance inst = instanceExample.instance;
							//for learner
							votes = learner.getVotesForInstance(inst);
							evaluator.addResult(instanceExample, votes);
						}					
						accumulatedDisc=0;	
					}
				}
				if (numberSamples==windowSize||(numberSamples-windowSize)%10==0){		
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					int[] rankedLabels = evaluator.getAucEstimator().getsortedLabels();																		
					Curve curve = new Curve(rankedLabels);
					double prArea = curve.prArea();
					if (accumulatedDisc==0)
						writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+","+favPos+","+favNeg+","+savPos+","+savNeg);
					else 
						writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+",1,1,1,1");
				}
			}
			
		}
		writer.close();
		System.out.println("done YYY");
	}
	//2.(YYN) Reset Learner + Continue Training + Use weight 1 after Reweighing
	public static void Reweighing_ResetContTrainingSameWeight(String filename, String outfile) throws FileNotFoundException{
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
		writer.println("noSamples, DiscData_YYN, DiscClassifier_YYN, Acc_YYN, P_YYN, R_YYN, F1_YYN, rocArea_YYN, prArea_YYN,favPos_YYN,favNeg_YYN,savPos_YYN,savNeg_YYN");
		windowList = new InstanceExample[windowSize];
		int numberSamples=0;
		double accumulatedDisc=0;double avgDisc = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			
			learner.trainOnInstance(trainInst);

			String[] labels;
			double[] predictions;
			int[] trueLabels;
			if (numberSamples>=windowSize){
				//for calculation of discrimination score
				labels = evaluator.getAucEstimator().getSAVal();
				predictions = evaluator.getAucEstimator().getPredictions();
				trueLabels = evaluator.getAucEstimator().getTrueLabel();
				double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
				double discData = Disc_Data();
				accumulatedDisc+=discClassifier;
				if (numberSamples==windowSize)
					avgDisc=discClassifier;
				else if ((numberSamples-windowSize)%averageSize==0)
					avgDisc=accumulatedDisc/averageSize;
				
				if ((numberSamples==windowSize)||((numberSamples-windowSize)%averageSize==0)){
					if (Math.abs(avgDisc)>epsilon){
						//reweighing
						int[] posWindow = evaluator.getAucEstimator().getPosWindowFromSortedScores();
						
						learner.resetLearning();
						ApplyReweighing(posWindow,labels, trueLabels);	
						//re-evaluate
						for (int i=0; i<windowSize; i++){
							InstanceExample instanceExample = windowList[i];
							Instance inst = instanceExample.instance;
							//for learner
							votes = learner.getVotesForInstance(inst);
							evaluator.addResult(instanceExample, votes);
						}					
						accumulatedDisc=0;	
					}
				}
				if (numberSamples==windowSize||(numberSamples-windowSize)%10==0){		
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					int[] rankedLabels = evaluator.getAucEstimator().getsortedLabels();																		
					Curve curve = new Curve(rankedLabels);
					double prArea = curve.prArea();
					if (accumulatedDisc==0)
						writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+","+favPos+","+favNeg+","+savPos+","+savNeg);
					else 
						writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+",1,1,1,1");
				}
			}
			
		}
		writer.close();
		System.out.println("done YYN");
	}		
	//3. (NYY) No Reset + Continue Training + Apply New Weight after Reweighing
	public static void Reweighing_NoResetContTrainingDiffWeight(String filename, String outfile) throws FileNotFoundException{
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
		writer.println("noSamples, DiscData_NYY, DiscClassifier_NYY, Acc_NYY, P_NYY, R_NYY, F1_NYY, rocArea_NYY, prArea_NYY,favPos_NYY,favNeg_NYY,savPos_NYY,savNeg_NYY");
		windowList = new InstanceExample[windowSize];
		int numberSamples=0;
		double accumulatedDisc=0;double avgDisc = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			
			if (favPos!=0){						
				String[] splits=trainInst.toString().split(",");
				int cl=Integer.parseInt(splits[splits.length-1]);
				if (splits[saIndex].equals(saVal)){//Deprived					
					if (cl==desiredClass)//Positive class
						trainInst.setWeight(savPos);
					else
						trainInst.setWeight(savNeg);
				}else{
					if (cl==desiredClass)//Positive class
						trainInst.setWeight(favPos);
					else
						trainInst.setWeight(favNeg);
				}
			}
			learner.trainOnInstance(trainInst);			
			
			String[] labels;
			double[] predictions;
			int[] trueLabels;
			if (numberSamples>=windowSize){
				//for calculation of discrimination score
				labels = evaluator.getAucEstimator().getSAVal();
				predictions = evaluator.getAucEstimator().getPredictions();
				trueLabels = evaluator.getAucEstimator().getTrueLabel();
				double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
				double discData = Disc_Data();
				accumulatedDisc+=discClassifier;
				if (numberSamples==windowSize)
					avgDisc=discClassifier;
				else if ((numberSamples-windowSize)%averageSize==0)
					avgDisc=accumulatedDisc/averageSize;
				
				if ((numberSamples==windowSize)||((numberSamples-windowSize)%averageSize==0)){
					if (Math.abs(avgDisc)>epsilon){
						//reweighing
						int[] posWindow = evaluator.getAucEstimator().getPosWindowFromSortedScores();
						
//						learner.resetLearning();
						ApplyReweighing(posWindow,labels, trueLabels);	
						//re-evaluate
						for (int i=0; i<windowSize; i++){
							InstanceExample instanceExample = windowList[i];
							Instance inst = instanceExample.instance;
							//for learner
							votes = learner.getVotesForInstance(inst);
							evaluator.addResult(instanceExample, votes);
						}					
						accumulatedDisc=0;	
					}
				}
				if (numberSamples==windowSize||(numberSamples-windowSize)%10==0){		
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					int[] rankedLabels = evaluator.getAucEstimator().getsortedLabels();																		
					Curve curve = new Curve(rankedLabels);
					double prArea = curve.prArea();
					if (accumulatedDisc==0)
						writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+","+favPos+","+favNeg+","+savPos+","+savNeg);
					else 
						writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+",1,1,1,1");
				}
			}
			
		}
		writer.close();
		System.out.println("done NYY");
	}
	//4. (NYN) No Reset + Continue Training + Use weight 1 after Reweighing
	public static void Reweighing_NoResetContTrainingSameWeight(String filename, String outfile) throws FileNotFoundException{
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
		writer.println("noSamples, DiscData_NYN, DiscClassifier_NYN, Acc_NYN, P_NYN, R_NYN, F1_NYN, rocArea_NYN, prArea_NYN,favPos_NYN,favNeg_NYN,savPos_NYN,savNeg_NYN");
		windowList = new InstanceExample[windowSize];
		int numberSamples=0;
		double accumulatedDisc=0;double avgDisc = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
				
			learner.trainOnInstance(trainInst);
			String[] labels;
			double[] predictions;
			int[] trueLabels;
			if (numberSamples>=windowSize){
				//for calculation of discrimination score
				labels = evaluator.getAucEstimator().getSAVal();
				predictions = evaluator.getAucEstimator().getPredictions();
				trueLabels = evaluator.getAucEstimator().getTrueLabel();
				double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
				double discData = Disc_Data();
				accumulatedDisc+=discClassifier;
				if (numberSamples==windowSize)
					avgDisc=discClassifier;
				else if ((numberSamples-windowSize)%averageSize==0)
					avgDisc=accumulatedDisc/averageSize;
					
				if ((numberSamples==windowSize)||((numberSamples-windowSize)%averageSize==0)){
					if (Math.abs(avgDisc)>epsilon){
						//reweighing
						int[] posWindow = evaluator.getAucEstimator().getPosWindowFromSortedScores();
							
//							learner.resetLearning();
						ApplyReweighing(posWindow,labels, trueLabels);	
						//re-evaluate
						for (int i=0; i<windowSize; i++){
							InstanceExample instanceExample = windowList[i];
							Instance inst = instanceExample.instance;
							//for learner
							votes = learner.getVotesForInstance(inst);
							evaluator.addResult(instanceExample, votes);
						}					
						accumulatedDisc=0;	
					}
				}
				if (numberSamples==windowSize||(numberSamples-windowSize)%10==0){		
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					int[] rankedLabels = evaluator.getAucEstimator().getsortedLabels();																		
					Curve curve = new Curve(rankedLabels);
					double prArea = curve.prArea();
					if (accumulatedDisc==0)
						writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+","+favPos+","+favNeg+","+savPos+","+savNeg);
					else 
						writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+",1,1,1,1");
				}
			}				
		}
		writer.close();
		System.out.println("done NYN");
	}
	//5. (YN_) Reset Learner + No Continue Training
	public static void Reweighing_ResetNoContTraining(String filename, String outfile) throws FileNotFoundException{
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
		writer.println("noSamples, DiscData_YN_, DiscClassifier_YN_, Acc_YN_, P_YN_, R_YN_, F1_YN_, rocArea_YN_, prArea_YN_,favPos_YN_,favNeg_YN_,savPos_YN_,savNeg_YN_");
		windowList = new InstanceExample[windowSize];
		int numberSamples=0;
		double accumulatedDisc=0;double avgDisc = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (numberSamples<=windowSize)
				learner.trainOnInstance(trainInst);
			
			String[] labels;
			double[] predictions;
			int[] trueLabels;
			if (numberSamples>=windowSize){
				//for calculation of discrimination score
				labels = evaluator.getAucEstimator().getSAVal();
				predictions = evaluator.getAucEstimator().getPredictions();
				trueLabels = evaluator.getAucEstimator().getTrueLabel();
				double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
				double discData = Disc_Data();
				accumulatedDisc+=discClassifier;
				if (numberSamples==windowSize)
					avgDisc=discClassifier;
				else if ((numberSamples-windowSize)%averageSize==0)
					avgDisc=accumulatedDisc/averageSize;
				
				if ((numberSamples==windowSize)||((numberSamples-windowSize)%averageSize==0)){
					if (Math.abs(avgDisc)>epsilon){
						//reweighing
						int[] posWindow = evaluator.getAucEstimator().getPosWindowFromSortedScores();
						
						learner.resetLearning();
						ApplyReweighing(posWindow,labels, trueLabels);	
						//re-evaluate
						for (int i=0; i<windowSize; i++){
							InstanceExample instanceExample = windowList[i];
							Instance inst = instanceExample.instance;
							//for learner
							votes = learner.getVotesForInstance(inst);
							evaluator.addResult(instanceExample, votes);
						}					
						accumulatedDisc=0;	
					}
				}
				if (numberSamples==windowSize||(numberSamples-windowSize)%10==0){		
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					int[] rankedLabels = evaluator.getAucEstimator().getsortedLabels();																		
					Curve curve = new Curve(rankedLabels);
					double prArea = curve.prArea();
					if (accumulatedDisc==0)
						writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+","+favPos+","+favNeg+","+savPos+","+savNeg);
					else 
						writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+",1,1,1,1");
				}
			}
			
		}
		writer.close();
		System.out.println("done YN_");
	}
	//6. (NN_) No Reset + No Continue Training
	public static void Reweighing_NoResetNoContTraining(String filename, String outfile) throws FileNotFoundException{
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
		writer.println("noSamples, DiscData_NN_, DiscClassifier_NN_, Acc_NN_, P_NN_, R_NN_, F1_NN_, rocArea_NN_, prArea_NN_,favPos_NN_,favNeg_NN_,savPos_NN_,savNeg_NN_");
		windowList = new InstanceExample[windowSize];
		int numberSamples=0;
		double accumulatedDisc=0;double avgDisc = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (numberSamples<=windowSize)
				learner.trainOnInstance(trainInst);
			
			String[] labels;
			double[] predictions;
			int[] trueLabels;
			if (numberSamples>=windowSize){
				//for calculation of discrimination score
				labels = evaluator.getAucEstimator().getSAVal();
				predictions = evaluator.getAucEstimator().getPredictions();
				trueLabels = evaluator.getAucEstimator().getTrueLabel();
				double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
				double discData = Disc_Data();
				accumulatedDisc+=discClassifier;
				if (numberSamples==windowSize)
					avgDisc=discClassifier;
				else if ((numberSamples-windowSize)%averageSize==0)
					avgDisc=accumulatedDisc/averageSize;
				
				if ((numberSamples==windowSize)||((numberSamples-windowSize)%averageSize==0)){
					if (Math.abs(avgDisc)>epsilon){
						//reweighing
						int[] posWindow = evaluator.getAucEstimator().getPosWindowFromSortedScores();
						
//						learner.resetLearning();
						ApplyReweighing(posWindow,labels, trueLabels);	
						//re-evaluate
						for (int i=0; i<windowSize; i++){
							InstanceExample instanceExample = windowList[i];
							Instance inst = instanceExample.instance;
							//for learner
							votes = learner.getVotesForInstance(inst);
							evaluator.addResult(instanceExample, votes);
						}					
						accumulatedDisc=0;	
					}
				}
				if (numberSamples==windowSize||(numberSamples-windowSize)%10==0){		
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					int[] rankedLabels = evaluator.getAucEstimator().getsortedLabels();																		
					Curve curve = new Curve(rankedLabels);
					double prArea = curve.prArea();
					if (accumulatedDisc==0)
						writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+","+favPos+","+favNeg+","+savPos+","+savNeg);
					else 
						writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+",1,1,1,1");
				}
			}
			
		}
		writer.close();
		System.out.println("done NN_");
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
		if ((nSaPos+nSaNeg)>0 && (saPos+saNeg)>0)
			return 100*((double)(tpFavored+fpFavored)/(double)(nSaPos+nSaNeg)
					-(double)(tpDeprived+fpDeprived)/(double)(saPos+saNeg));
		else
			return -100;//denominator==0
	}	
	
	public static double Disc_Data(){//DiscriminationScore should be calculated first
		saPos=tpDeprived+fnDeprived;
		saNeg=tnDeprived+fpDeprived;
		nSaPos=tpFavored+fnFavored;		
		nSaNeg=tnFavored+fpFavored;		
		if ((nSaPos+nSaNeg)>0 && (saPos+saNeg)>0)
			return 100*((double)(nSaPos)/(double)(nSaPos+nSaNeg)
				-(double)(saPos)/(double)(saPos+saNeg));
		else
			return -100;
	}

	public static void ApplyReweighing(int[] posWindow, String[] labels, int[] trueLabels){
		//weight calculation
		if (saPos!=0)
			savPos=(double)(saPos+saNeg)*(double)(saPos+nSaPos)/(double)(windowSize*saPos);
		else
			savPos=1;
		if (saNeg!=0)
			savNeg=(double)(saPos+saNeg)*(double)(saNeg+nSaNeg)/(double)(windowSize*saNeg);
		else
			savNeg=1;
		if (nSaPos!=0)
			favPos=(double)(nSaPos+nSaNeg)*(double)(saPos+nSaPos)/(double)(windowSize*nSaPos);
		else
			favPos=1;
		if (nSaNeg!=0)
			favNeg=(double)(nSaPos+nSaNeg)*(double)(saNeg+nSaNeg)/(double)(windowSize*nSaNeg);
		else
			favNeg=1;
		//apply new weight for the current window
		for (int i=0; i<windowSize; i++){
			if (labels[i].equals(saVal)){ //Deprived
				if (trueLabels[i]==desiredClass){//positive
					windowList[posWindow[i]%windowSize].instance.setWeight(savPos);
				}else{
					windowList[posWindow[i]%windowSize].instance.setWeight(savNeg);
				}
			}else{//Favored
				if (trueLabels[i]==desiredClass){//positive
					windowList[posWindow[i]%windowSize].instance.setWeight(favPos);
				}else{
					windowList[posWindow[i]%windowSize].instance.setWeight(favNeg);
				}
			}
		}//end for		

		//retrain the model
		for (int i=0; i<windowSize; i++){
			learner.trainOnInstance(windowList[i].instance);
		}
	}
}
