package geniusweb.exampleparties.kayseriliagent;


import geniusweb.issuevalue.Value;

import java.util.*;

public class OpponentModelling {
    private Collection<Value> issueValueList;
    private Double totalOccurrence;
    private ArrayList<Collection<Value>> issuesList = new ArrayList<Collection<Value>>();
    private HashMap<String, HashMap<String, Double>> bidsHistory;
    private ArrayList<Double> opponentSpaceList;

    public OpponentModelling(){
        opponentSpaceList = new ArrayList<>();
    }

    public void addOpponentModel(Collection<Value> issueValueList) {
        this.issueValueList = issueValueList;
        this.issuesList.add(issueValueList);


    }
    public void printBids(){
        for (Collection<Value> issue : issuesList){
            //System.out.println(issue + " : " + this.totalOccurrence);
        }
        calculateFrequency();
    }
    public ArrayList<Double> calculateFrequency(){
        HashMap<String,Double> values = new HashMap<>();
        ArrayList<HashMap<String,Double>> valuesList = new ArrayList<HashMap<String,Double>>();
        double sum = 0.0;
        int size = 0;

        for (String k : bidsHistory.keySet()) {
            if (k != null)
                values = bidsHistory.get(k);
            valuesList.add(values);
        }
        for (Collection<Value> issue: issuesList){
            size += issue.size();
            for (int i = 0; i<issue.size(); i++){
                String issueVal = issue.toArray()[i].toString();
                for (HashMap<String,Double> tmpValues: valuesList){
                    if (tmpValues.containsKey(issueVal)){
                        sum += tmpValues.get(issueVal);
                    }
                }
                double sumNum = sum / size / this.totalOccurrence;
                opponentSpaceList.add(sumNum);

            }
        }
            return opponentSpaceList;
    }

    public void validateOthers(Double turnCount, HashMap<String, HashMap<String, Double>> bidsHistory) {
        this.bidsHistory = bidsHistory;
        this.totalOccurrence = turnCount;
    }
}

