package ar.uba.fi.celdas;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Theories {
	
	Map<Integer,List<Theory>> theories;
	Set<Integer> existenceSet;
	
	public Theories(){
		this.theories = new HashMap<Integer, List<Theory>>();
		this.existenceSet = new HashSet<Integer>();
	}
	
	public void add(Theory theory) throws Exception{
		if(!existsTheory(theory)){			
			List<Theory> theoryList = this.theories.get(theory.hashCodeOnlyCurrentState());
			if(theoryList == null){
				theoryList = new ArrayList<Theory>();
			}
			theoryList.add(theory);
			this.existenceSet.add(theory.hashCode());
		}else{
			throw new Exception("Theory already exist!");
		}
	}
	
	public boolean existsTheory(Theory theory){		
		return this.existenceSet.contains(theory.hashCode());
	}
	
	public List<Theory> getSortedListForCurrentState(Theory theory){
		
		List<Theory> theoryList = this.theories.get(theory.hashCodeOnlyCurrentState());
		if(theoryList == null){
			theoryList = new ArrayList<Theory>();
		}		
		Collections.sort(theoryList);
		return theoryList;
	}

	public Set<Integer> getExistenceSet() {
		return existenceSet;
	}

	public void setExistenceSet(Set<Integer> existenceSet) {
		this.existenceSet = existenceSet;
	}

	public Map<Integer, List<Theory>> getTheories() {
		return theories;
	}

	public void setTheories(Map<Integer, List<Theory>> theories) {
		this.theories = theories;
	}

}
