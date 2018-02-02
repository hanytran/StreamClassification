package source;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.InstanceExample;
//import moa.evaluation.AdwinClassificationPerformanceEvaluator;
//import moa.evaluation.WindowClassificationPerformanceEvaluator;
import moa.streams.ArffFileStream;
import moa.streams.generators.RandomRBFGenerator;
//import weka.core.Instances;


public class Massaging {
	
	protected static int saPos = 0, saNeg=0, nSaPos = 0, nSaNeg=0;
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
	protected static ArrayList<Instance> windowList = new ArrayList<Instance>();
	protected static int windowSize=0;
	
	protected static double[][] sortedPromotionList;
	protected static double[][] sortedDemotionList;
	
	protected static int numCorrectSAPos = 0;
	protected static int numCorrectSANeg = 0;
	protected static int numCorrectNsaPos = 0;
	protected static int numCorrectNsaNeg = 0;
	
	protected static int numWrongSAPos = 0;//SA: Pos=>classified=>Neg
	protected static int numWrongSANeg =0;//SA: Neg=>classified=>Pos
	protected static int numWrongNsaPos = 0;
	protected static int numWrongNsaNeg = 0;
	
	protected static int averageSize = 0;
	
	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub		
//		String filename = "adult_stream.arff";//before remove fnlwgt attribute
		String outfile = "new_dataset_for_han_removefnlwgt_stream.arff";
//		remove_fnlwgt(filename,outfile);
//		checkData(outfile);
//		staticMassaging(outfile);
		
		windowSize 	= 1000;
		averageSize = 1000;
		saName = "sex";
		saVal = "Female";		
		massaging(outfile);
			
	}
	public static void checkData(String filename) {
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
		
		//evaluation: accuracy
		int numberSamplesCorrect = 0;
		int numberSamples = 0;
		
		numCorrectSAPos = 0;
		numCorrectSANeg = 0;
		numCorrectNsaPos = 0;
		numCorrectNsaNeg = 0;
		
		numWrongSAPos = 0;//SA: Pos=>classified=>Neg
		numWrongSANeg =0;//SA: Neg=>classified=>Pos
		numWrongNsaPos = 0;
		numWrongNsaNeg = 0;	
		int SAPos = 0, SANeg=0, NsaPos=0, NsaNeg=0;
		
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
//			double a = trainInstance.getData().classValue();
			String[] splits = trainInst.toString().split(",");
			String sa = splits[saIndex];
			int classVal = Integer.parseInt(splits[splits.length-1]);
			//check data
			if (sa.equals(saVal)&&classVal==desiredClass)
				SAPos++;
			else if (sa.equals(saVal)&&classVal==notDesiredClass)
				SANeg++;
			else if (!sa.equals(saVal)&&classVal==desiredClass)
				NsaPos++;
			else
				NsaNeg++;
			//classification
			if (learner.correctlyClassifies(trainInst) && classVal==desiredClass){//positive class
				if (sa.equals(saVal))//SA
					numCorrectSAPos++;
				else
					numCorrectNsaPos++;									
			}else if (learner.correctlyClassifies(trainInst) && classVal==notDesiredClass){
				if (sa.equals(saVal))//SA
					numCorrectSANeg++; 
				else
					numCorrectNsaNeg++;
			}else if (classVal==desiredClass) {//Pos => classified => Neg
				if (sa.equals(saVal))//SA
					numWrongSAPos++;
				else
					numWrongNsaPos++;
			}else{ //Neg=>classified=>Pos
				if (sa.equals(saVal))//SA
					numWrongSANeg++;
				else
					numWrongNsaNeg++;
			}
			//accuracy in total
			if (learner.correctlyClassifies(trainInst)){				
				numberSamplesCorrect++;
			}
			learner.trainOnInstance(trainInst);
			numberSamples++;
		}
		double accuracy = 100*(double)numberSamplesCorrect/(double)numberSamples;				
		System.out.println("Done");	
	}
	
	public static void staticMassaging (String filename){
		windowList = new ArrayList<Instance>();//reset windowList
		ArffFileStream fs = new ArffFileStream(filename, -1);		
		//get saIndex from the stream header
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}		
		fs.prepareForUse();
						
		int numberSamplesCorrect = 0;
		int numberSamples = 0;
		
		numCorrectSAPos = 0;
		numCorrectSANeg = 0;
		numCorrectNsaPos = 0;
		numCorrectNsaNeg = 0;
		
		numWrongSAPos = 0;//SA: Pos=>classified=>Neg
		numWrongSANeg =0;//SA: Neg=>classified=>Pos
		numWrongNsaPos = 0;
		numWrongNsaNeg = 0;
		
		double accuracy = 0;
//		boolean isTesting = false;
		
		ranker.setModelContext(fs.getHeader());
		ranker.prepareForUse();
		while (fs.hasMoreInstances()){			
			Instance trainInst = fs.nextInstance().instance;			
			windowList.add(numberSamples, trainInst);
			//train for ranker
			if (ranker.correctlyClassifies(trainInst))
				numberSamplesCorrect++;
			ranker.trainOnInstance(trainInst);
		}
		//massaging
		double discriminationScore = groupNum();
		if (discriminationScore>epsilon){
			double changes = discriminationScore*(double)(saPos+saNeg)*(double)(nSaPos+nSaNeg)
					/((double)windowList.size()*100);
			int numRelables = ranking_relabel(discriminationScore, changes);
			System.out.println(numRelables);
		}
		//train the model with massaged data
		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();
//		learner.resetLearning();
		int correctTraining = 0;
		for (int i=0; i<windowList.size(); i++){
			if (learner.correctlyClassifies(windowList.get(i)))
				correctTraining++;
			learner.trainOnInstance(windowList.get(i));
		}
		accuracy = 100*(double)correctTraining/(double)windowList.size();
		System.out.println(accuracy);
		
		fs.restart();
		numberSamplesCorrect=0;
		while (fs.hasMoreInstances()){
			Instance trainInst = fs.nextInstance().instance;
			String[] splits = trainInst.toString().split(",");
			String sa = splits[saIndex];
			int classVal = Integer.parseInt(splits[splits.length-1]);
			
			if (learner.correctlyClassifies(trainInst) && classVal==desiredClass){//positive class
				if (sa.equals(saVal))//SA
					numCorrectSAPos++;
				else
					numCorrectNsaPos++;
				numberSamplesCorrect++;
			}else if (learner.correctlyClassifies(trainInst) && classVal==notDesiredClass){
				if (sa.equals(saVal))//SA
					numCorrectSANeg++;
				else
					numCorrectNsaNeg++;
				numberSamplesCorrect++;
			}else if (classVal==desiredClass) {//Pos => classified => Neg
				if (sa.equals(saVal))//SA
					numWrongSAPos++;
				else
					numWrongNsaPos++;
			}else{ //Neg=>classified=>Pos
				if (sa.equals(saVal))//SA
					numWrongSANeg++;
				else
					numWrongNsaNeg++;
			}				
			//accuracy in total
//			if (learner.correctlyClassifies(trainInst)){				
//				numberSamplesCorrect++;
//			}
			numberSamples++;
		}
		accuracy = 100*(double)numberSamplesCorrect/(double)numberSamples;
		System.out.println("Done");
	}
	
	public static void massaging(String filename) throws FileNotFoundException{	
		ArffFileStream fs = new ArffFileStream(filename, -1);		
		saIndex = 0;
		//get saIndex from the stream header
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}			
		fs.prepareForUse();						
		//prepare to use
		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();
		
		//windowEvaluation definition
//		WindowClassificationPerformanceEvaluator evaluator = new WindowClassificationPerformanceEvaluator();
//		evaluator.widthOption.setValue(windowSize);
//		evaluator.prepareForUse();
		
//		AdwinClassificationPerformanceEvaluator e2 = new AdwinClassificationPerformanceEvaluator();
//		e2.reset(2);
//		e2.prepareForUse();
		
		//evaluation: accuracy
		int numberSamplesCorrect = 0;
		int numberSamples = 0;
		//evaluation: F1
		numCorrectSAPos = 0;
		numCorrectSANeg = 0;
		numCorrectNsaPos = 0;
		numCorrectNsaNeg = 0;
		
		numWrongSAPos = 0;//SA: Pos=>classified=>Neg
		numWrongSANeg =0;//SA: Neg=>classified=>Pos
		numWrongNsaPos = 0;
		numWrongNsaNeg = 0;
		
		boolean isTesting = false;
				
		double discrimnationScore = 0;
		double accuracy = 0;
		int numRetrain = 0;
		
		int numChanges=0;
		double accumulateDisc=0;
		int cnt=0;
		while (fs.hasMoreInstances()){
			//1st window enough
			if (numberSamples>=windowSize)
				isTesting = true;
			Instance trainInst = fs.nextInstance().instance;			
			String[] splits = trainInst.toString().split(",");
			String sa = splits[saIndex];
			int classVal = Integer.parseInt(splits[splits.length-1]);
			
			if (learner.correctlyClassifies(trainInst) && classVal==desiredClass){//positive class
				if (sa.equals(saVal))//SA
					numCorrectSAPos++;
				else
					numCorrectNsaPos++;		
				numberSamplesCorrect++;
			}else if (learner.correctlyClassifies(trainInst) && classVal==notDesiredClass){
				if (sa.equals(saVal))//SA
					numCorrectSANeg++;
				else
					numCorrectNsaNeg++;
				numberSamplesCorrect++;
			}else if (classVal==desiredClass) {//Pos => classified => Neg
				if (sa.equals(saVal))//SA
					numWrongSAPos++;
				else
					numWrongNsaPos++;
			}else{ //Neg=>classified=>Pos
				if (sa.equals(saVal))//SA
					numWrongSANeg++;
				else
					numWrongNsaNeg++;
			}
			//accuracy in total
//			if (learner.correctlyClassifies(trainInst)){				
//				numberSamplesCorrect++;
//			}
			/*******************
			 * no. of instances < windowSize: test then train instances
			 ********************/
			if (!isTesting){				
				windowList.add(numberSamples, trainInst); //add instances into 1st window								
				learner.trainOnInstance(trainInst);											
			}else{			
				/********************
				compute no. of SaPos, SaNeg, NSaPos, NSaNeg
				check discrimination score
				 *********************/				 
				discrimnationScore = groupNum();
				accumulateDisc+=discrimnationScore;
				cnt++;
				if ((numberSamples-windowSize)%averageSize==0){
					double averageDisc = 0;
					if (numberSamples == (windowSize))//1st window
						averageDisc = accumulateDisc;
					else
						averageDisc = accumulateDisc/averageSize;
					System.out.println(averageDisc);
					accumulateDisc=0;
					cnt=0;
					if (averageDisc>epsilon){
						//train the ranker
						ranker.setModelContext(fs.getHeader());
						ranker.prepareForUse();					
						for (Instance inst:windowList){													
							//train for ranker
							ranker.correctlyClassifies(inst);
							ranker.trainOnInstance(inst);
						}
						//**massaging & retrain the model**/
						//massaging
						numRetrain++;
						double changes = discrimnationScore*(double)(saPos+saNeg)*(double)(nSaPos+nSaNeg)/((double)windowList.size()*100);
						numChanges+=ranking_relabel(averageDisc, changes);
						discrimnationScore = groupNum();
						//retrain the model
						learner.resetLearning();
						for (Instance inst:windowList){
							learner.correctlyClassifies(inst);
							learner.trainOnInstance(inst);
						}				
						//evaluator window
	//					double[] votes = learner.getVotesForInstance(trainInst);
	//					evaluator.addResult(trainInstanceExample, votes);					
					}
				}
				slideWindow(trainInst);				
			}
			numberSamples++;						
		}		
//		writer.close();
		accuracy = 100*(double)numberSamplesCorrect/(double)numberSamples;
		System.out.println("number of Changes: "+ numChanges);
		System.out.println("number of retrain: "+ numRetrain);
		System.out.println(numberSamples+" instances processed with accuracy "+accuracy);	
	}

	public static int ranking_relabel(double discrimnationScore,
			double changes){
		double[][] promotionList=new double[saNeg][2];
	    double[][] demotionList=new double[nSaPos][2];
	    int demote = 0, promote = 0;
	    for (int i=0; i<windowList.size(); i++){
			Instance inst = windowList.get(i);
			String[] splits = inst.toString().split(",");
			String sa = splits[saIndex];
			int classVal = Integer.parseInt(splits[splits.length-1]);
			
			if (sa.equals(saVal) && (classVal!=desiredClass)){
				double prob = ranker.getVotesForInstance(inst)[1];
				promotionList[promote][0]=i;
				promotionList[promote][1]=prob;
				promote++;
			}else if (!sa.equals(saVal) && (classVal==desiredClass)){
				double prob = ranker.getVotesForInstance(inst)[0];
				demotionList[demote][0]=i;
				demotionList[demote][1]=prob;
				demote++;
			}		
		}//end of for i
	    sortedPromotionList = sorting(promotionList, saNeg, 1);
	    sortedDemotionList = sorting(demotionList, nSaPos, 2);
	    int i=0;
	    while (discrimnationScore>epsilon){//loop until get expected Discrimination score
//			for (int i=1; i<changes; i++){
	    	int index = 0;
				index = (int)sortedPromotionList[i][0];				
				windowList.get(index).setClassValue(desiredClass);
//			}
//			for (int i=1; i<changes; i++){
				index = (int)sortedDemotionList[i][0];
				windowList.get(index).setClassValue(notDesiredClass);
//			}	
				discrimnationScore = groupNum();
				i++;
		}
	    return i;
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
	
	public static double groupNum(){
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;
		for (Instance inst:windowList){
			String[] splits = inst.toString().split(",");
			String sa = splits[saIndex];
			int classVal = Integer.parseInt(splits[splits.length-1]);
			if (sa.equals(saVal)){
				if (classVal==desiredClass)
					saPos++;
				else
					saNeg++;
			}else{
				if (classVal==desiredClass)
					nSaPos++;
				else
					nSaNeg++;
			}
		}
		return 100*((double)nSaPos/(double)(nSaPos+nSaNeg)-(double)saPos/(double)(saPos+saNeg));
	}
	
    public static void slideWindow(Instance comingInst){
    	windowList.remove(0);
    	windowList.add(windowSize-1, comingInst);
    }
		
	public static void remove_fnlwgt(String filenameIn, String filenameOut) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(filenameIn, -1);		
		PrintWriter writer = new PrintWriter(filenameOut);
		writer.write(fs.getHeader().toString());
		while (fs.hasMoreInstances()){
			Instance inst = fs.nextInstance().instance;
			String[] splits = inst.toString().split(",");			
			writer.write(splits[0]+", "+splits[1]+", "+splits[3]+", "+splits[4]+", "+
					splits[5]+", "+splits[6]+", "+splits[7]+", "+splits[8]+", "+splits[9]+", "+
					splits[10]+", "+splits[11]+", "+splits[12]+", "+splits[13]+", "+splits[14]+"\n");
		}
		writer.close();
	}
}
