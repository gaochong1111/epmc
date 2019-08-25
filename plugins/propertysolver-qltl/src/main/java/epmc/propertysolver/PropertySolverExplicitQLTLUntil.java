/****************************************************************************

    ePMC - an extensible probabilistic model checker
    Copyright (C) 2017

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

 *****************************************************************************/

package epmc.propertysolver;

import java.util.LinkedHashSet;
import java.util.Set;

import epmc.expression.Expression;
import epmc.expression.standard.DirType;
import epmc.expression.standard.ExpressionQuantifier;
import epmc.graph.CommonProperties;
import epmc.graph.StateMap;
import epmc.graph.StateSet;
import epmc.graph.UtilGraph;
import epmc.graph.explicit.GraphExplicit;
import epmc.graph.explicit.StateSetExplicit;
import epmc.modelchecker.EngineExplicit;
import epmc.modelchecker.ModelChecker;
import epmc.modelchecker.PropertySolver;
import epmc.operator.Operator;
import epmc.value.TypeArray;
import epmc.value.TypeWeight;
import epmc.value.UtilValue;
import epmc.value.ValueArray;

public final class PropertySolverExplicitQLTLUntil implements PropertySolver {
    public final static String IDENTIFIER = "qltl-explicit";
    private ModelChecker modelChecker;
    private GraphExplicit graph;
    private StateSetExplicit computeForStates;
    private Expression property;
    private StateSet forStates;

    @Override
    public void setModelChecker(ModelChecker modelChecker) {
        assert modelChecker != null;
        this.modelChecker = modelChecker;
        if (modelChecker.getEngine() instanceof EngineExplicit) {
            this.graph = modelChecker.getLowLevel();
        }
    }

    @Override
    public void setProperty(Expression property) {
        this.property = property;
    }

    @Override
    public void setForStates(StateSet forStates) {
        this.forStates = forStates;
    }

    @Override
    public StateMap solve() {
        assert property != null;
        assert forStates != null;
        assert property instanceof ExpressionQuantifier;
        StateSetExplicit forStatesExplicit = (StateSetExplicit) forStates;
        
        //////
        System.out.println("QLTL: ");
        System.out.println(graph);
        
        System.exit(0);
        
        graph.explore(forStatesExplicit.getStatesExplicit());
        ExpressionQuantifier propertyQuantifier = (ExpressionQuantifier) property;
        Expression quantifiedProp = propertyQuantifier.getQuantified();
        DirType dirType = ExpressionQuantifier.computeQuantifierDirType(propertyQuantifier);
        StateMap result = doSolve(quantifiedProp, forStates, dirType.isMin());
        
        
        if (!propertyQuantifier.getCompareType().isIs()) {
            StateMap compare = modelChecker.check(propertyQuantifier.getCompare(), forStates);
            Operator op = propertyQuantifier.getCompareType().asExOpType();
            assert op != null;
            result = result.applyWith(op, compare);
        }
        return result;
    }

    private StateMap doSolve(Expression property, StateSet states, boolean min) {
        ///// 
        /// System.out.println(graph);
        return null;
    }

    
    @Override
    public boolean canHandle() {
        assert property != null;
        if (!(modelChecker.getEngine() instanceof EngineExplicit)) {
            return false;
        }
//        Semantics semantics = modelChecker.getModel().getSemantics();
//        if (!SemanticsDiscreteTime.isDiscreteTime(semantics)
//                && !SemanticsContinuousTime.isContinuousTime(semantics)) {
//            return false;
//        }
//        if (!ExpressionQuantifier.is(property)) {
//            return false;
//        }
        // handleSimplePCTLExtensions();
//        ExpressionQuantifier propertyQuantifier = ExpressionQuantifier.as(property);
//        if (!UtilPCTL.isPCTLPathUntil(propertyQuantifier.getQuantified())) {
//            return false;
//        }
//        Set<Expression> inners = UtilPCTL.collectPCTLInner(propertyQuantifier.getQuantified());
//        GraphExplicit graph = modelChecker.getLowLevel();
//        System.out.println(graph);
//        StateSet allStates = UtilGraph.computeAllStatesExplicit(graph);
//        System.exit(0);
//        for (Expression inner : inners) {
//            modelChecker.ensureCanHandle(inner, allStates);
//        }
//        if (allStates != null) {
//            allStates.close();
//        }
        return true;
    }

    
    @Override
    public Set<Object> getRequiredGraphProperties() {
        Set<Object> required = new LinkedHashSet<>();
        required.add(CommonProperties.SEMANTICS);
        return required;
    }

    @Override
    public Set<Object> getRequiredNodeProperties() {
        Set<Object> required = new LinkedHashSet<>();
        required.add(CommonProperties.STATE);
        required.add(CommonProperties.PLAYER);
        ExpressionQuantifier propertyQuantifier = (ExpressionQuantifier) property;
        Set<Expression> inners = UtilQLTL.collectQLTLInner(propertyQuantifier.getQuantified());
        StateSet allStates = UtilGraph.computeAllStatesExplicit(modelChecker.getLowLevel());
        for (Expression inner : inners) {
            required.addAll(modelChecker.getRequiredNodeProperties(inner, allStates));
        }
        return required;
    }

    @Override
    public Set<Object> getRequiredEdgeProperties() {
        Set<Object> required = new LinkedHashSet<>();
        required.add(CommonProperties.WEIGHT);
        return required;
    }

    @Override
    public String getIdentifier() {
        return IDENTIFIER;
    }

    private ValueArray newValueArrayWeight(int size) {
        TypeArray typeArray = TypeWeight.get().getTypeArray();
        return UtilValue.newArray(typeArray, size);
    }


}
