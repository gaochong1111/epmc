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

package epmc.jani;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import epmc.error.EPMCException;
import epmc.jani.dd.ProblemsJANIDD;
import epmc.jani.model.ModelJANI;
import epmc.jani.model.OptionsJANIModel;
import epmc.main.options.UtilOptionsEPMC;
import epmc.modelchecker.EngineDD;
import epmc.modelchecker.ExploreStatistics;
import epmc.modelchecker.Model;
import epmc.modelchecker.options.OptionsModelChecker;
import epmc.options.Options;
import epmc.plugin.OptionsPlugin;

import static epmc.graph.TestHelperGraph.*;
import static epmc.jani.ModelNames.*;
import static epmc.modelchecker.TestHelper.*;

import java.math.BigInteger;

/**
 * Tests for symbolic DD-based state exploration of JANI models.
 * 
 * @author Ernst Moritz Hahn
 */
public final class ExploreDDTest {
    /** Location of plugin directory in file system. */
    private final static String PLUGIN_DIR = System.getProperty("user.dir") + "/target/classes/";

    /**
     * Set up the tests.
     */
    @BeforeClass
    public static void initialise() {
        prepare();
    }

    /**
     * Prepare options including loading JANI plugin.
     * 
     * @return options usable for JANI model analysis
     */
    private final static Options prepareJANIOptions() {
        Options options = UtilOptionsEPMC.newOptions();
        options.set(OptionsPlugin.PLUGIN, PLUGIN_DIR);
        prepareOptions(options, ModelJANI.IDENTIFIER);
        return options;
    }

    /**
     * Test to explore a minimal JANI model.
     * 
     */
    @Test
    public void minimalTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        options.set(OptionsJANIModel.JANI_FIX_DEADLOCKS, false);
        Model model = null;
        boolean thrown = false;
        try {
            model = loadModel(options, MINIMAL);
            exploreModel(model);
        } catch (EPMCException e) {
            thrown = true;
            Assert.assertEquals(ProblemsJANIDD.JANI_DD_DEADLOCK, e.getProblem());
        }
        Assert.assertTrue(thrown);		
    }

    /**
     * Test to explore a minimal non-deadlock MDP JANI model.
     * 
     */
    @Test
    public void minimalNonDeadlockMDPTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        ExploreStatistics statistics = null;
        model = loadModel(options, MINIMAL_NON_DEADLOCK_MDP);
        statistics = exploreModel(model);
        Assert.assertEquals(new BigInteger("1"), statistics.getNumNodes());
        Assert.assertEquals(new BigInteger("1"), statistics.getNumStates());
        Assert.assertEquals(new BigInteger("1"), statistics.getNumTransitions());
    }

    /**
     * Test to explore a minimal non-deadlock DTMC JANI model.
     * 
     */
    @Test
    public void minimalNonDeadlockDTMCTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        ExploreStatistics statistics = null;
        model = loadModel(options, MINIMAL_NON_DEADLOCK_DTMC);
        statistics = exploreModel(model);
        Assert.assertEquals(new BigInteger("1"), statistics.getNumNodes());
        Assert.assertEquals(new BigInteger("1"), statistics.getNumStates());
        Assert.assertEquals(new BigInteger("1"), statistics.getNumTransitions());
    }

    /**
     * Test to explore an MDP model of a cycle of two states.
     * 
     */
    @Test
    public void twoStatesCycleMDPTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        ExploreStatistics statistics = null;
        model = loadModel(options, TWO_STATES_CYCLE_MDP);
        statistics = exploreModel(model);
        Assert.assertEquals(new BigInteger("2"), statistics.getNumNodes());
        Assert.assertEquals(new BigInteger("2"), statistics.getNumStates());
        Assert.assertEquals(new BigInteger("2"), statistics.getNumTransitions());
    }

    /**
     * Test to explore a DTMC model of a cycle of two states.
     * 
     */
    @Test
    public void twoStatesCycleDTMCPTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        ExploreStatistics statistics = null;
        model = loadModel(options, TWO_STATES_CYCLE_DTMC);
        statistics = exploreModel(model);
        Assert.assertEquals(new BigInteger("2"), statistics.getNumNodes());
        Assert.assertEquals(new BigInteger("2"), statistics.getNumStates());
        Assert.assertEquals(new BigInteger("2"), statistics.getNumTransitions());
    }

    /**
     * Test to explore Knuth's dice model.
     * 
     */
    @Test
    public void diceTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        model = loadModel(options, DICE);
        ExploreStatistics statistics = exploreModel(model);
        Assert.assertEquals(new BigInteger("13"), statistics.getNumNodes());
        Assert.assertEquals(new BigInteger("13"), statistics.getNumStates());
        Assert.assertEquals(new BigInteger("20"), statistics.getNumTransitions());
    }

    /**
     * Test to explore Knuth's dice model with global variable.
     * 
     */
    @Test
    public void diceGlobalTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        model = loadModel(options, DICE_GLOBAL);
        ExploreStatistics statistics = exploreModel(model);
        Assert.assertEquals(new BigInteger("13"), statistics.getNumNodes());
        Assert.assertEquals(new BigInteger("13"), statistics.getNumStates());
        Assert.assertEquals(new BigInteger("20"), statistics.getNumTransitions());
    }

    /**
     * Test of cell model.
     * 
     */
    @Test
    public void cellTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        model = loadModel(options, CELL);
        ExploreStatistics statistics = exploreModel(model);
        Assert.assertEquals(new BigInteger("7"), statistics.getNumNodes());
        Assert.assertEquals(new BigInteger("7"), statistics.getNumStates());
        Assert.assertEquals(new BigInteger("12"), statistics.getNumTransitions());
    }

    /**
     * Test of MDP diamond model.
     * 
     */
    @Test
    public void diamondMDPTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        model = loadModel(options, DIAMOND_MDP);
        ExploreStatistics statistics = exploreModel(model);
        Assert.assertEquals(new BigInteger("4"), statistics.getNumNodes());
        Assert.assertEquals(new BigInteger("4"), statistics.getNumStates());
        Assert.assertEquals(new BigInteger("7"), statistics.getNumTransitions());
    }

    /**
     * Test of CTMC diamond model.
     * 
     */
    @Test
    public void diamondCTMCTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        model = loadModel(options, DIAMOND_CTMC);
        ExploreStatistics statistics = exploreModel(model);
        Assert.assertEquals(new BigInteger("4"), statistics.getNumNodes());
        Assert.assertEquals(new BigInteger("4"), statistics.getNumStates());
        Assert.assertEquals(new BigInteger("7"), statistics.getNumTransitions());
    }

    /**
     * MDP test whether simple synchronisation works.
     * 
     */
    @Test
    public void syncMDPTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        model = loadModel(options, SYNC_MDP);
        ExploreStatistics statistics = exploreModel(model);
        Assert.assertEquals(new BigInteger("2"), statistics.getNumNodes());
        Assert.assertEquals(new BigInteger("2"), statistics.getNumStates());
        Assert.assertEquals(new BigInteger("2"), statistics.getNumTransitions());
    }

    /**
     * DTMC test whether simple synchronisation works.
     * 
     */
    @Test
    public void syncDTMCTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        model = loadModel(options, SYNC_DTMC);
        ExploreStatistics statistics = exploreModel(model);
        Assert.assertEquals(new BigInteger("2"), statistics.getNumNodes());
        Assert.assertEquals(new BigInteger("2"), statistics.getNumStates());
        Assert.assertEquals(new BigInteger("2"), statistics.getNumTransitions());
    }

    /**
     * Conflict writing to global variable.
     * 
     */
    @Test
    public void syncConflictTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        model = loadModel(options, SYNC_CONFLICT);
        boolean thrown = false;
        try {
            exploreModel(model);        	
        } catch (EPMCException e) {
            thrown = true;
            Assert.assertEquals(ProblemsJANIDD.JANI_DD_GLOBAL_MULTIPLE, e.getProblem());
        }
        Assert.assertTrue(thrown);
    }

    /**
     * Test for rewards.
     * 
     */
    @Test
    public void rewardsTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        model = loadModel(options, REWARDS);
        System.out.println(exploreModelGraph(model));
    }

    /**
     * Test for BEB model from Arnd Hartmanns.
     * 
     */
    @Test
    public void bebTest() {
        Options options = prepareJANIOptions();
        prepareOptions(options, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelJANI.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineDD.class);
        Model model = null;
        model = loadModel(options, BEB);
        System.out.println(model);
        ExploreStatistics statistics = exploreModel(model);
        Assert.assertEquals(new BigInteger("4660"), statistics.getNumNodes());
        Assert.assertEquals(new BigInteger("4660"), statistics.getNumStates());
        Assert.assertEquals(new BigInteger("7031"), statistics.getNumTransitions());
    }    
}
