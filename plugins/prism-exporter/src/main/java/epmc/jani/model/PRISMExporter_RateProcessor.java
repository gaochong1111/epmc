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

package epmc.jani.model;

import epmc.prism.exporter.processor.PRISMExporter_ProcessorStrict;
import epmc.prism.exporter.processor.PRISMExporter_ProcessorRegistrar;

public class PRISMExporter_RateProcessor implements PRISMExporter_ProcessorStrict {

    private Rate rate = null;

    @Override
    public PRISMExporter_ProcessorStrict setElement(Object obj) {
        assert obj != null;
        assert obj instanceof Rate; 

        rate = (Rate) obj;
        return this;
    }

    @Override
    public String toPRISM() {
        assert rate != null;

        return PRISMExporter_ProcessorRegistrar.getProcessor(rate.getExp())
                .toPRISM();
    }

    @Override
    public void validateTransientVariables() {
        assert rate != null;

        PRISMExporter_ProcessorRegistrar.getProcessor(rate.getExp())
            .validateTransientVariables();
    }

    @Override
    public boolean usesTransientVariables() {
        assert rate != null;

        return PRISMExporter_ProcessorRegistrar.getProcessor(rate.getExp())
                .usesTransientVariables();
    }	
}
