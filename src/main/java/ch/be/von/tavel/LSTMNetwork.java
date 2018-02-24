package ch.be.von.tavel;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class LSTMNetwork {

    private static final Logger LOG = LoggerFactory.getLogger(LSTMNetwork.class);

    private final MultiLayerNetwork net;

    private LSTMNetwork(MultiLayerNetwork net) {
        this.net = net;
    }

    public static LSTMNetwork create(int numberOfInputColumns, int lstmLayerSize, int numberOfOutputs, int tbpttLength, int randomSeed) {
        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.1)
                .seed(randomSeed)
                .regularization(true)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(numberOfInputColumns).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX) //MCXENT + softmax for classification
                        .nIn(lstmLayerSize).nOut(numberOfOutputs).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return new LSTMNetwork(net);
    }

    public static LSTMNetwork restore(String path) throws IOException {
        LOG.info("Restoring model from {}", path);
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(path);
        return new LSTMNetwork(net);
    }

    public void save(String path) throws IOException {
        LOG.info("Saving model to {}", path);
        ModelSerializer.writeModel(net, path, true);
    }

    public void setListeners(IterationListener... listeners) {
        this.net.setListeners(listeners);
    }

    public void fit(DataSet data) {
        net.fit(data);
    }

    public INDArray rnnTimeStep(INDArray input) {
        return net.rnnTimeStep(input);
    }

    public void rnnClearPreviousState() {
        net.rnnClearPreviousState();
    }

    public Layer[] getLayers() {
        return net.getLayers();
    }

}
