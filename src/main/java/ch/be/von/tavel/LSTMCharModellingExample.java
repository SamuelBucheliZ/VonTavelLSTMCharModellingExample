package ch.be.von.tavel;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class LSTMCharModellingExample {

    private static final Logger LOG = LoggerFactory.getLogger(LSTMCharModellingExample.class);

    private final Configuration config;
    private final Sampler sampler;

    public LSTMCharModellingExample(Configuration config) {
        this.config = config;
        this.sampler = new Sampler(config.getRandomSeed(), config.getnCharactersToSample(), config.getnSamplesToGenerate());
    }

    public static void main(String[] args) throws Exception {
        String textSource = LSTMCharModellingExample.class.getClassLoader().getResource("von_tavel.txt").getFile();
        String modelSource = LSTMCharModellingExample.class.getClassLoader().getResource("model.zip").getFile();
        Configuration config = Configuration.builder()
                .setTextSource(textSource)
                .setModelFileInPath(modelSource)
                .setValidCharacters(null)
                .build();
        LSTMCharModellingExample lstmCharModellingExample = new LSTMCharModellingExample(config);
        lstmCharModellingExample.run();
    }

    public void run() throws IOException {
        FileLoader fileLoader = new FileLoader(config.getMiniBatchSize(), config.getExampleLength(), config.getRandomSeed(), config.getValidCharacters());

        //Get a DataSetIterator that handles vectorization of text into something we can use to train
        // our GravesLSTM network.
        CharacterIterator iter = fileLoader.getIterator(config.getTextSource());
        int numberOfOutputs = iter.totalOutcomes();

        //Restore or set up network configuration:
        LSTMNetwork net;
        if (config.getModelFileInPath() == null) {
            net = LSTMNetwork.create(iter.inputColumns(), config.getLstmLayerSize(), numberOfOutputs, config.getTbpttLength(), config.getRandomSeed());
        } else {
            net = LSTMNetwork.restore(config.getModelFileInPath());
        }

        // for some nice UI and output, see http://localhost:9000/train
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));

        //Print the  number of parameters in the network (and for each layer)
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            int nParams = layers[i].numParams();
            LOG.info("Number of parameters in layer {}: {}", i, nParams);
            totalNumParams += nParams;
        }
        LOG.info("Total number of network parameters: {}", totalNumParams);

        train(iter, net);

        LOG.info("\n\nExample complete");

        net.save(config.getModelFileOutPath());

        LOG.info("--------------------");
        LOG.info("Final Sample");

        logSample(iter, net);
    }

    private void train(CharacterIterator iter, LSTMNetwork net) {
        //Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
        for (int i = 0; i < config.getNumEpochs(); i++) {
            for (DataSet ds : iter.toIterable()) {
                net.fit(ds);

                if (++miniBatchNumber % config.getGenerateSamplesEveryNMinibatches() == 0) {
                    logSampleForBatch(iter, net, miniBatchNumber);
                }
            }
            iter.reset();    //Reset iterator for another epoch
        }

    }

    private void logSampleForBatch(CharacterIterator iter, LSTMNetwork net, int miniBatchNumber) {
        LOG.info("--------------------");
        LOG.info("Completed {} minibatches of size {}x{} characters", miniBatchNumber, config.getMiniBatchSize(), config.getExampleLength());
        logSample(iter, net);
    }

    private void logSample(CharacterIterator iter, LSTMNetwork net) {
        LOG.info("Sampling characters from network given initialization \"{}\"", config.getGenerationInitialization() == null ? "" : config.getGenerationInitialization());
        String[] samples = sampler.sampleCharactersFromNetwork(config.getGenerationInitialization(), net, iter);
        for (int j = 0; j < samples.length; j++) {
            LOG.info("----- Sample {} -----", j);
            LOG.info(samples[j]);
            LOG.info("-----");
        }
    }

}

