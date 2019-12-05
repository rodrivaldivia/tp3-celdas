package ar.uba.fi.celdas;

import java.awt.event.KeyEvent;

import ontology.Types;
import ontology.Types.ACTIONS;
import tools.Vector2d;

public class Theory  implements Comparable<Theory>{

	private char[][] currentState = null;
	private Types.ACTIONS action;
	private char[][] predictedState;
	int usedCount;
	int successCount;
	float utility;
	
	public char[][] getCurrentState() {
		return currentState;
	}
	public void setCurrentState(char[][] currentState) {
		this.currentState = currentState;
	}
	public Types.ACTIONS getAction() {
		return action;
	}
	public void setAction(Types.ACTIONS action) {
		this.action = action;
	}
	public char[][] getPredictedState() {
		return predictedState;
	}
	public void setPredictedState(char[][] predictedState) {
		this.predictedState = predictedState;
	}
	public int getUsedCount() {
		return usedCount;
	}
	public void setUsedCount(int usedCount) {
		this.usedCount = usedCount;
	}
	public int getSuccessCount() {
		return successCount;
	}
	public void setSuccessCount(int successCount) {
		this.successCount = successCount;
	}
	public float getUtility() {
		return utility;
	}
	public void setUtility(float utility) {
		this.utility = utility;
	}
	
	private String charArrayToStr(char[][] charrarray ){
		StringBuilder sb = new StringBuilder("");
		if(charrarray!=null){
			 for(int i=0;i< charrarray.length; i++){
		        	for(int j=0;j<  charrarray[i].length; j++){
		        		sb.append(charrarray[i][j]);
		        	}
		        	sb.append("\n");
			 }
		}
		return sb.toString();
	}
	
	public String actionToString(){
		switch(this.action){
			case ACTION_NIL: return "ACTION_NIL";
			case ACTION_UP: return "ACTION_UP";
			case ACTION_LEFT: return "ACTION_LEFT";
			case ACTION_DOWN: return "ACTION_DOWN";
			case ACTION_RIGHT: return "ACTION_RIGHT";
			case ACTION_USE: return "ACTION_USE";
			case ACTION_ESCAPE: return "ACTION_ESCAPE";
		}
		return "";
	}
	
	public String toString(){
		StringBuilder sb = new StringBuilder("");
		sb.append(this.charArrayToStr(currentState));
		sb.append("\n");
		sb.append(this.actionToString());
		sb.append("\n");
		sb.append(this.charArrayToStr(predictedState));		
		return sb.toString();
	}
	
	@Override
	public int hashCode() {
		
		return this.toString().hashCode();
	}
	
	public int hashCodeOnlyCurrentState() {
		return this.charArrayToStr(currentState).hashCode();
	}

   @Override
   public boolean equals(Object obj) {
      if (this == obj)
         return true;
      if (obj == null)
         return false;
      if (getClass() != obj.getClass())
         return false;
      Theory other = (Theory) obj;
      if (!this.toString().equals(other.toString()))
         return false;
      return true;
   }
  
	@Override
	public int compareTo(Theory o) {
		if(this.utility == o.utility){
			float sucessThis = (float) this.successCount / this.usedCount;
			float sucessOther = (float) o.successCount / o.usedCount;
			return (int)Math.round((sucessThis-sucessOther)*100);
		}
		return (int)Math.round((this.utility-o.utility)*100);
	}
	
}

