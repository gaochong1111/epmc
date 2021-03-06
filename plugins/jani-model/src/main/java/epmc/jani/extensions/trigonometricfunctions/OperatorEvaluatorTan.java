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

package epmc.jani.extensions.trigonometricfunctions;

import epmc.operator.Operator;
import epmc.value.OperatorEvaluator;
import epmc.value.Type;
import epmc.value.TypeDouble;
import epmc.value.TypeInteger;
import epmc.value.TypeReal;
import epmc.value.Value;
import epmc.value.ValueDouble;
import epmc.value.ValueNumber;
import epmc.value.operatorevaluator.OperatorEvaluatorSimpleBuilder;

public final class OperatorEvaluatorTan implements OperatorEvaluator {
    public final static class Builder implements OperatorEvaluatorSimpleBuilder {
        private boolean built;
        private Operator operator;
        private Type[] types;

        @Override
        public void setOperator(Operator operator) {
            assert !built;
            this.operator = operator;
        }

        @Override
        public void setTypes(Type[] types) {
            assert !built;
            this.types = types;
        }

        @Override
        public OperatorEvaluator build() {
            assert !built;
            assert operator != null;
            assert types != null;
            built = true;
            for (Type type : types) {
                assert type != null;
            }
            if (operator !=  OperatorTan.TAN) {
                return null;
            }
            if (types.length != 1) {
                return null;
            }
            if (!TypeDouble.is(types[0]) && !TypeInteger.is(types[0])) {
                return null;
            }
            return new OperatorEvaluatorTan(this);
        }
    }

    private OperatorEvaluatorTan(Builder builder) {
    }

    @Override
    public void apply(Value result, Value... operands) {
        assert result != null;
        assert operands != null;
        assert operands.length >= 1;
        assert operands[0] != null;
        ValueDouble resultDouble = ValueDouble.as(result);
        ValueNumber operandNumber = ValueNumber.as(operands[0]);
        resultDouble.set(Math.tan(operandNumber.getDouble()));
    }

    @Override
    public Type resultType() {
        return TypeReal.get();
    }
}
