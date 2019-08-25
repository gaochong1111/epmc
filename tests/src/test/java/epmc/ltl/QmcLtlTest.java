package epmc.ltl;

import static epmc.modelchecker.TestHelper.assertEquals;
import static epmc.modelchecker.TestHelper.computeResult;
import static epmc.modelchecker.TestHelper.prepare;
import static epmc.modelchecker.TestHelper.prepareOptions;

import java.util.ArrayList;
import java.util.List;

import org.junit.BeforeClass;
import org.junit.Test;

import epmc.main.options.UtilOptionsEPMC;
import epmc.modelchecker.EngineExplicit;
import epmc.modelchecker.TestHelper;
import epmc.modelchecker.options.OptionsModelChecker;
import epmc.operator.OperatorSet;
import epmc.options.Options;
import epmc.plugin.OptionsPlugin;
import epmc.qmc.model.ModelPRISMQMC;
import epmc.qmc.model.PropertyPRISMQMC;
import epmc.qmc.value.TypeMatrix;
import epmc.qmc.value.ValueMatrix;
import epmc.value.ContextValue;
import epmc.value.OperatorEvaluator;
import epmc.value.TypeInteger;
import epmc.value.TypeReal;
import epmc.value.Value;
import epmc.value.ValueInteger;

/*
* Tests for model checking of QMC with LTL properties.
* 
* @author Chong Gao
*/
public class QmcLtlTest {
    /** Location of plugin directory in file system. */
    //    private final static String PLUGIN_DIR = System.getProperty("user.dir") + "/target/classes/";

    /**
     * Set up the tests.
     */
    @BeforeClass
    public static void initialise() {
        prepare();
    }

    
    private final static Options prepareQMCOptions() {
        List<String> qmcPlugins = new ArrayList<>();
        qmcPlugins.add(System.getProperty("user.dir") + "/plugins/qmc/target/classes/");
        //qmcPlugins.add(System.getProperty("user.dir") + "/plugins/qmc-exporter/target/classes/");
        
        Options options = UtilOptionsEPMC.newOptions();
        options.set(OptionsPlugin.PLUGIN, qmcPlugins);
        prepareOptions(options, ModelPRISMQMC.IDENTIFIER);
        options.set(OptionsModelChecker.MODEL_INPUT_TYPE, ModelPRISMQMC.IDENTIFIER);
        options.set(OptionsModelChecker.PROPERTY_INPUT_TYPE, PropertyPRISMQMC.IDENTIFIER);
        options.set(OptionsModelChecker.ENGINE, EngineExplicit.class);
        return options;
    }
    
    private final static String PREFIX = System.getProperty("user.home") + "/work/test/";
    public final static String DICE_MODEL = PREFIX + "test.prism";
    public final static String DICE_PROPERTY = PREFIX + "test.props";
    public final static String LOOP_MODEL = PREFIX + "qmc-loop.prism";
    public final static String QMC_PROPERTY = PREFIX + "qmc-loop.props";
    

    
    @Test
    public void loopTest() {
        Options options = prepareQMCOptions();
        double tolerance = 1E-10;
        options.set(TestHelper.ITERATION_TOLERANCE, Double.toString(tolerance));
        Value result1 = computeResult(options, LOOP_MODEL, "Q>=1 [ F (s=3) ]");
        TypeMatrix typeArray = TypeMatrix.get(TypeReal.get());
        ValueMatrix compare23 = typeArray.newValue();
        compare23.setDimensions(2, 2);
        set(compare23, 1, 0, 0);
        assertEquals(true, result1);
//        Value result2 = computeResult(options, LOOP_MODEL, "qeval(Q=?[F (s=3)], |p>_2 <p|_2)");
//        assertEquals(compare23, result2, tolerance);
//        Value result3 = computeResult(options, LOOP_MODEL, "qeval(Q=?[F (s=3)], ID(2)/2)");
//        assertEquals(compare23, result3, tolerance);
    }
    
    private static void set(ValueMatrix valueArray, int entry, int row, int col) {
        Value valueEntry = valueArray.getType().getEntryType().newValue();
        ValueInteger valueInt = TypeInteger.get().newValue();
        valueInt.set(entry);
        OperatorEvaluator set = ContextValue.get().getEvaluator(OperatorSet.SET, TypeInteger.get(), valueArray.getType().getEntryType());
        set.apply(valueEntry, valueInt);
        valueArray.set(valueEntry, row, col);
    }
    
   
}
