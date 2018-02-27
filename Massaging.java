package source;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
//import moa.classifiers.trees.HoeffdingTree;
import moa.core.InstanceExample;
import moa.evaluation.WindowAUCImbalancedPerformanceEvaluator;
import moa.streams.ArffFileStream;



public class Massaging {
	
	protected static int saPos = 0, saNeg=0, nSaPos = 0, nSaNeg=0;
	protected static int tpDeprived=0, fpDeprived=0, tpFavored=0, fpFavored=0; 
	protected static int tnDeprived=0, fnDeprived=0, tnFavored=0, fnFavored=0;
	
	protected static int epsilon = 5; //percentage
	
	//different classifiers
	protected static Classifier learner = new NaiveBayes();
	protected static Classifier ranker = new NaiveBayes();
//	protected static Classifier learner = new HoeffdingTree();
//	protected static Classifier learner= new HoeffdingAdaptiveTree();
	//data definition
	protected static int saIndex=0;
	protected static String saVal = "Female";
	protected static int desiredClass = 1;
	protected static int notDesiredClass = 0;
	
	protected static String saName = "sex";
	protected static int windowSize=0;
	protected static InstanceExample[] windowList = new InstanceExample[windowSize];
//	protected static ArrayList<String> classifiedLabel=new ArrayList<String>();
	
	
	protected static double[][] sortedPromotionList;
	protected static double[][] sortedDemotionList;
	protected static double[][] sortedList=null;
	protected static double[][] problist;
		
	protected static int averageSize = 0;

	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub		
		windowSize 	= 1000;
		averageSize = 500;
		saName = "sex";
		saVal = "Female";		
//		String filename = "adult_stream.arff";//before remove fnlwgt attribute
		String infile = "new_dataset_for_han_removefnlwgt_stream.arff";
//		String out="out/original_1K.csv";
//		originalStreamClassification(infile, out);

		String out = "out/500_1K_NN_NB.csv";
//		Massaging_ResetContTrain(infile, out);
//		Massaging_ResetNoContTrain(infile, out);
//		Massaging_NoResetContTrain(infile, out);
		Massaging_NoResetNoContTrain(infile, out);
		System.out.println("main done");			
	}	
	public static void Massaging_ResetContTrain(String filename, String outfile) 
			throws FileNotFoundException{		
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
		ranker.setModelContext(fs.getHeader());
		ranker.prepareForUse();	
		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();
		//evaluator for ranker
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();
		
		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData_YY, DiscClassifier_YY,changes_YY, Acc_YY, P_YY, R_YY, F1_YY, rocArea_YY, prArea_YY");//, prArea");		
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0;
		double accumulatedDisc=0;double avgDisc = 0;double changes=0;
		windowList = new InstanceExample[windowSize];
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);			
//			double kappaStatistic = evaluator.getKappaStatistic();

			double[] ranker_votes = ranker.getVotesForInstance(trainInstanceExample);
			ranker_evaluator.addResult(trainInstanceExample, ranker_votes);
			
			learner.trainOnInstance(trainInst);
			ranker.trainOnInstance(trainInst);
			
			String[] labels;
			double[] predictions;
			int[] trueLabels;
			if (numberSamples>=windowSize)	{		
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
						//massaging
						int saClassified=0, nSaClassified=0;
						int saNum =0, nSaNum=0;
						saClassified=tpDeprived+fpDeprived;
						nSaClassified=tpFavored+fpFavored;
						saNum = saPos+saNeg;
						nSaNum= nSaPos+nSaNeg;						
						if (avgDisc<-epsilon){//reverse discrimination & previous did the change							
							if (saVal.equals("Female"))
								saVal="Male";
							else
								saVal="Female";
//							changes=((double)(saPos)*(double)nSaNum-(double)(nSaPos)*(double)saNum
//									+(double)(epsilon/100)*(double)saNum*(double)nSaNum)
//									/(double)(windowSize);
							changes=((double)saClassified*(double)nSaNum-(double)nSaClassified*(double)saNum
									-(double)(epsilon/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}else{
//							changes=((double)(nSaPos)*(double)saNum-(double)(saPos)*(double)nSaNum
//									-(double)(epsilon/100)*(double)saNum*(double)nSaNum)
//									/(double)(windowSize);
							changes=((double)nSaClassified*(double)saNum-(double)saClassified*(double)nSaNum
									-(double)(epsilon/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}
						if (changes>0){
//						numRetrain++;
						//massaging
						int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
						int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
						String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
						double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
						rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
						relabel_M(changes);
							
						learner.resetLearning();
						ranker.resetLearning();
						retrainModel(learner);
						retrainModel(ranker);
						//re-evaluate
						for (int i=0; i<windowSize; i++){
							InstanceExample instanceExample = windowList[i];
							Instance inst = instanceExample.instance;
							//for learner
							votes = learner.getVotesForInstance(inst);
							evaluator.addResult(instanceExample, votes);
							//for ranker
							ranker_votes = ranker.getVotesForInstance(instanceExample);
							ranker_evaluator.addResult(instanceExample, ranker_votes);
						}
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
//					double holdoutAUC=evaluator.getAucEstimator().getHoldoutAUC();
//					double ratio = evaluator.getAucEstimator().getRatio();
					int[] rankedLabels = evaluator.getAucEstimator().getsortedLabels();																		
					Curve curve = new Curve(rankedLabels);
					double prArea = curve.prArea();
//					double rocArea = curve.rocArea();
					
					//for calculation of discrimination score: after
//					labels = evaluator.getAucEstimator().getSAVal();
//					predictions = evaluator.getAucEstimator().getPredictions();
//					trueLabels = evaluator.getAucEstimator().getTrueLabel();
//					discClassifier=DiscriminationScore(labels,predictions,trueLabels);
//					discData = Disc_Data();
					if (accumulatedDisc==0)
						writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea);
					else 
						writer.println(numberSamples+","+discData+","+discClassifier+","+"0"+","+acc+","
								+precision+","+recall+","+f1+","+rocArea+","+prArea);
				}
			}
		}
		writer.close();
		System.out.println("Done");	
	}
	
	public static void Massaging_ResetNoContTrain(String filename, String outfile) 
			throws FileNotFoundException{				
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
		ranker.setModelContext(fs.getHeader());
		ranker.prepareForUse();	
		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();
		//evaluator for ranker
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();
		
		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData_YN, DiscClassifier_YN,changes_YN, Acc_YN, P_YN, R_YN, F1_YN, rocArea_YN, prArea_YN");//, prArea");		
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0;
		double accumulatedDisc=0;double avgDisc = 0;double changes=0;
		windowList = new InstanceExample[windowSize];
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples%windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);			
//			double kappaStatistic = evaluator.getKappaStatistic();

			double[] ranker_votes = ranker.getVotesForInstance(trainInstanceExample);
			ranker_evaluator.addResult(trainInstanceExample, ranker_votes);
			//NO continue training
			if (numberSamples<=windowSize)	{
				learner.trainOnInstance(trainInst);
				ranker.trainOnInstance(trainInst);
			}
			String[] labels;
			double[] predictions;
			int[] trueLabels;
			if (numberSamples>=windowSize)	{		
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
						//massaging
						int saClassified=0, nSaClassified=0;
						int saNum =0, nSaNum=0;
						saClassified=tpDeprived+fpDeprived;
						nSaClassified=tpFavored+fpFavored;
						saNum = saPos+saNeg;
						nSaNum= nSaPos+nSaNeg;						
						if (avgDisc<-epsilon){//reverse discrimination							
							if (saVal=="Female")
								saVal="Male";
							else
								saVal="Female";
							changes=((double)saClassified*(double)nSaNum-(double)nSaClassified*(double)saNum
									-(double)(epsilon/100)*(double)saNum*(double)nSaNum)
									/(double)(saNum+nSaNum);
						}else{
							changes=((double)nSaClassified*(double)saNum-(double)saClassified*(double)nSaNum
									-(double)(epsilon/100)*(double)saNum*(double)nSaNum)
									/(double)(saNum+nSaNum);
						}
//						numRetrain++;
						//massaging
						if (changes>0){
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
								
							learner.resetLearning();
							ranker.resetLearning();
							retrainModel(learner);
							retrainModel(ranker);
							//re-evaluate
							for (int i=0; i<windowSize; i++){
								InstanceExample instanceExample = windowList[i];
								Instance inst = instanceExample.instance;
								//for learner
								votes = learner.getVotesForInstance(inst);
								evaluator.addResult(instanceExample, votes);
								//for ranker
								ranker_votes = ranker.getVotesForInstance(instanceExample);
								ranker_evaluator.addResult(instanceExample, ranker_votes);
							}
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
//					double holdoutAUC=evaluator.getAucEstimator().getHoldoutAUC();
//					double ratio = evaluator.getAucEstimator().getRatio();
					int[] rankedLabels = evaluator.getAucEstimator().getsortedLabels();																		
					Curve curve = new Curve(rankedLabels);
					double prArea = curve.prArea();
//					double rocArea = curve.rocArea();
					
					//for calculation of discrimination score
//					labels = evaluator.getAucEstimator().getSAVal();
//					predictions = evaluator.getAucEstimator().getPredictions();
//					trueLabels = evaluator.getAucEstimator().getTrueLabel();
//					discClassifier=DiscriminationScore(labels,predictions,trueLabels);
//					discData = Disc_Data();
					if (accumulatedDisc==0)
						writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea);
					else 
						writer.println(numberSamples+","+discData+","+discClassifier+","+"0"+","+acc+","
								+precision+","+recall+","+f1+","+rocArea+","+prArea);
				}
			}
		}
		writer.close();
		System.out.println("Done");	
	}
	
	public static void Massaging_NoResetContTrain(String filename, String outfile) 
			throws FileNotFoundException{		
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
		ranker.setModelContext(fs.getHeader());
		ranker.prepareForUse();	
		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();
		//evaluator for ranker
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();
		
		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData_NY, DiscClassifier_NY,changes_NY, Acc_NY, P_NY, R_NY, F1_NY, rocArea_NY, prArea_NY");//, prArea");		
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0;
		double accumulatedDisc=0;double avgDisc = 0;double changes=0;
		windowList = new InstanceExample[windowSize];
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples%windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);			
//			double kappaStatistic = evaluator.getKappaStatistic();

			double[] ranker_votes = ranker.getVotesForInstance(trainInstanceExample);
			ranker_evaluator.addResult(trainInstanceExample, ranker_votes);
			
			learner.trainOnInstance(trainInst);
			ranker.trainOnInstance(trainInst);

			String[] labels;
			double[] predictions;
			int[] trueLabels;
			if (numberSamples>=windowSize)	{		
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
						//massaging
						int saClassified=0, nSaClassified=0;
						int saNum =0, nSaNum=0;
						saClassified=tpDeprived+fpDeprived;
						nSaClassified=tpFavored+fpFavored;
						saNum = saPos+saNeg;
						nSaNum= nSaPos+nSaNeg;						
						if (avgDisc<-epsilon){//reverse discrimination							
							if (saVal=="Female")
								saVal="Male";
							else
								saVal="Female";
							changes=((double)saClassified*(double)nSaNum-(double)nSaClassified*(double)saNum
									-(double)(epsilon/100)*(double)saNum*(double)nSaNum)
									/(double)(saNum+nSaNum);
						}else{
							changes=((double)nSaClassified*(double)saNum-(double)saClassified*(double)nSaNum
									-(double)(epsilon/100)*(double)saNum*(double)nSaNum)
									/(double)(saNum+nSaNum);
						}
//						numRetrain++;
						//massaging
						if (changes>0){
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
								
	//						learner.resetLearning();
	//						ranker.resetLearning();
							retrainModel(learner);
							retrainModel(ranker);
							//re-evaluate
							for (int i=0; i<windowSize; i++){
								InstanceExample instanceExample = windowList[i];
								Instance inst = instanceExample.instance;
								//for learner
								votes = learner.getVotesForInstance(inst);
								evaluator.addResult(instanceExample, votes);
								//for ranker
								ranker_votes = ranker.getVotesForInstance(instanceExample);
								ranker_evaluator.addResult(instanceExample, ranker_votes);
							}
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
//					double holdoutAUC=evaluator.getAucEstimator().getHoldoutAUC();
//					double ratio = evaluator.getAucEstimator().getRatio();
					int[] rankedLabels = evaluator.getAucEstimator().getsortedLabels();																		
					Curve curve = new Curve(rankedLabels);
					double prArea = curve.prArea();
//					double rocArea = curve.rocArea();
					
					//for calculation of discrimination score
//					labels = evaluator.getAucEstimator().getSAVal();
//					predictions = evaluator.getAucEstimator().getPredictions();
//					trueLabels = evaluator.getAucEstimator().getTrueLabel();
//					discClassifier=DiscriminationScore(labels,predictions,trueLabels);
//					discData = Disc_Data();
					if (accumulatedDisc==0)
						writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea);
					else 
						writer.println(numberSamples+","+discData+","+discClassifier+","+"0"+","+acc+","
								+precision+","+recall+","+f1+","+rocArea+","+prArea);
				}
			}
		}
		writer.close();
		System.out.println("Done");	
	}
	
	public static void Massaging_NoResetNoContTrain(String filename, String outfile) 
			throws FileNotFoundException{		
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
		ranker.setModelContext(fs.getHeader());
		ranker.prepareForUse();	
		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();
		//evaluator for ranker
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();
		
		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData_NN, DiscClassifier_NN,changes_NN, Acc_NN, P_NN, R_NN, F1_NN, rocArea_NN, prArea_NN");//, prArea");		
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0;
		double accumulatedDisc=0;double avgDisc = 0;double changes=0;
		windowList = new InstanceExample[windowSize];
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples%windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);			
//			double kappaStatistic = evaluator.getKappaStatistic();

			double[] ranker_votes = ranker.getVotesForInstance(trainInstanceExample);
			ranker_evaluator.addResult(trainInstanceExample, ranker_votes);
			if (numberSamples<=windowSize)	{
				learner.trainOnInstance(trainInst);
				ranker.trainOnInstance(trainInst);
			}
			String[] labels;
			double[] predictions;
			int[] trueLabels;
			if (numberSamples>=windowSize)	{		
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
						//massaging
						int saClassified=0, nSaClassified=0;
						int saNum =0, nSaNum=0;
						saClassified=tpDeprived+fpDeprived;
						nSaClassified=tpFavored+fpFavored;
						saNum = saPos+saNeg;
						nSaNum= nSaPos+nSaNeg;						
						if (avgDisc<-epsilon){//reverse discrimination							
							if (saVal=="Female")
								saVal="Male";
							else
								saVal="Female";
							changes=((double)saClassified*(double)nSaNum-(double)nSaClassified*(double)saNum
									-(double)(epsilon/100)*(double)saNum*(double)nSaNum)
									/(double)(saNum+nSaNum);
						}else{
							changes=((double)nSaClassified*(double)saNum-(double)saClassified*(double)nSaNum
									-(double)(epsilon/100)*(double)saNum*(double)nSaNum)
									/(double)(saNum+nSaNum);
						}
//						numRetrain++;
						//massaging
						if (changes>0){
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
								
	//						learner.resetLearning();
	//						ranker.resetLearning();
							retrainModel(learner);
							retrainModel(ranker);
							//re-evaluate
							for (int i=0; i<windowSize; i++){
								InstanceExample instanceExample = windowList[i];
								Instance inst = instanceExample.instance;
								//for learner
								votes = learner.getVotesForInstance(inst);
								evaluator.addResult(instanceExample, votes);
								//for ranker
								ranker_votes = ranker.getVotesForInstance(instanceExample);
								ranker_evaluator.addResult(instanceExample, ranker_votes);
							}
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
//					double holdoutAUC=evaluator.getAucEstimator().getHoldoutAUC();
//					double ratio = evaluator.getAucEstimator().getRatio();
					int[] rankedLabels = evaluator.getAucEstimator().getsortedLabels();																		
					Curve curve = new Curve(rankedLabels);
					double prArea = curve.prArea();
//					double rocArea = curve.rocArea();
					
					//for calculation of discrimination score
//					labels = evaluator.getAucEstimator().getSAVal();
//					predictions = evaluator.getAucEstimator().getPredictions();
//					trueLabels = evaluator.getAucEstimator().getTrueLabel();
//					discClassifier=DiscriminationScore(labels,predictions,trueLabels);
//					discData = Disc_Data();
					if (accumulatedDisc==0)
						writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea);
					else 
						writer.println(numberSamples+","+discData+","+discClassifier+","+"0"+","+acc+","
								+precision+","+recall+","+f1+","+rocArea+","+prArea);
				}
			}
		}
		writer.close();
		System.out.println("Done");
	}
	
	public static void originalStreamClassification(String filename, String outfile) 
			throws FileNotFoundException{		
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
		writer.println("noSamples, DiscData, DiscClassifier, Acc, P, R, F1, rocArea, prArea");//, prArea");		
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			numberSamples++;									
						
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);			
//			double kappaStatistic = evaluator.getKappaStatistic();
			learner.trainOnInstance(trainInst);
					
			if (numberSamples>=windowSize)	{			
				if (numberSamples==windowSize||(numberSamples-windowSize)%10==0){					
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
//					double holdoutAUC=evaluator.getAucEstimator().getHoldoutAUC();
//					double ratio = evaluator.getAucEstimator().getRatio();
//					double scoredAUC=evaluator.getAucEstimator().getScoredAUC();
					int[] rankedLabels = evaluator.getAucEstimator().getsortedLabels();																		
					Curve curve = new Curve(rankedLabels);
					double prArea = curve.prArea();
//					double rocArea = curve.rocArea();
					
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
		System.out.println("Done");	
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
			
	public static void retrainModel(Classifier classifier){
		for (InstanceExample inst:windowList){
			classifier.trainOnInstance(inst.instance);
		}
	}
	
	public static void rankingWithSA(int[] posWindow, String[] saValFromSortedScores,int[] sortedLabels, double[] sortedScores){
		double[][] promotionList=new double[windowSize][2];
	    double[][] demotionList=new double[windowSize][2];
	    int demote = 0, promote = 0;
	    for (int i=0; i<posWindow.length; i++){
	    	String sa = saValFromSortedScores[i];
			int classVal = sortedLabels[i];			
			if (sa.equals(saVal) && classVal==notDesiredClass){					 
				promotionList[promote][0]=posWindow[i]%windowSize;
				promotionList[promote++][1]=sortedScores[i];
			}else if (!sa.equals(saVal) && classVal==desiredClass){
				demotionList[demote][0]=posWindow[i]%windowSize;
				demotionList[demote++][1]=sortedScores[i];
			}
		}//end of for i
	    sortedPromotionList = sorting(promotionList, promote, 1);
	    sortedDemotionList = sorting(demotionList, demote, 2);
	}
	
	public static void relabel_M(double changes){
	    for (int i=0; i<changes; i++){
	    	int index=0;
			index = (int)sortedPromotionList[i][0];				
			windowList[index].instance.setClassValue(desiredClass);

			index = (int)sortedDemotionList[i][0];
			windowList[index].instance.setClassValue(notDesiredClass);
		}
	}

	  /* method to sort the 2-D arrays
	  * @param arrayToSort A 2-D array which we want to sort
	  * @param type 1 is descending order and type 2 is for ascending order
	  * @return sorted array
	  */
	  public static double[][] sorting(double [][] arrayToSort,int length,int type){
		  int max=length;
	      double val1=0,val2=0;
	      double [][]sortedArray=new double[length][2];
	      double [][] temp=new double[1][2];
	      for(int index=0;index<length;index++)  
	    	  for(int i=0;i<max-1;i++){
	    		  try{  
	                 val1=arrayToSort[i][1];
	                 val2=arrayToSort[i+1][1];
	        
	                        if(val1<val2 && type==1){  //swapping for sort descending
	                                 System.arraycopy(arrayToSort[i],0,temp[0],0,2);
	                                 System.arraycopy(arrayToSort[i+1],0,arrayToSort[i],0,2);
	                                 System.arraycopy(temp[0],0,arrayToSort[i+1],0,2);//System.out.println("val1 = "+val1+" new value of rec[] "+rec[i+1][20]+" i = "+i);
	                          }     //end of  if
	                            else if(val1>val2 && type==2){  //swapping for sort ascending
	                                 System.arraycopy(arrayToSort[i],0,temp[0],0,2);
	                                 System.arraycopy(arrayToSort[i+1],0,arrayToSort[i],0,2);
	                                 System.arraycopy(temp[0],0,arrayToSort[i+1],0,2);//System.out.println("val1 = "+val1+" new value of rec[] "+rec[i+1][20]+" i = "+i);
	                          }     //end of else if 
	             
	               } catch (NumberFormatException e){
	                 System.out.println(" Probelme with sorting during Massaging");
	               }
	               
	            }//end of out for-i loop
	             for(int i=0;i<length;i++)
	             System.arraycopy(arrayToSort[i],0, sortedArray[i],0, 2);
	             return sortedArray;
	}   // End of sorting function
	
}
