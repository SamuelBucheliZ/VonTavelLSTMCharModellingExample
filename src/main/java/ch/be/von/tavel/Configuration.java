package ch.be.von.tavel;

public class Configuration {

    private final int lstmLayerSize;
    private final int miniBatchSize;
    private final int exampleLength;
    private final int tbpttLength;
    private final int numEpochs;
    private final int generateSamplesEveryNMinibatches;
    private final int nSamplesToGenerate;
    private final int nCharactersToSample;
    private final String generationInitialization;
    private final int randomSeed;
    private final String textSource;
    private final char[] validCharacters;
    private final String modelFileOutPath;
    private final String modelFileInPath;


    /**
     * @param lstmLayerSize                    Number of units in each GravesLSTM layer
     * @param miniBatchSize                    Size of mini batch to use when training
     * @param exampleLength                    Length of each training example sequence to use. This could certainly be increased
     * @param tbpttLength                      Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
     * @param numEpochs                        Total number of training epochs
     * @param generateSamplesEveryNMinibatches How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
     * @param nSamplesToGenerate               Number of samples to generate after each training epoch
     * @param nCharactersToSample              Length of each sample to generate
     * @param generationInitialization         Optional character initialization; a random character is used if null. It is used to 'prime' the LSTM with a character sequence to continue/complete. Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
     * @param randomSeed
     * @param textSource
     * @param validCharacters                  Which characters are allowed? Others will be removed. If null, all characters are allowed.
     * @param modelFileOutPath                 Where to save the model file to
     * @param modelFileInPath                  Where to load the model file from (to continue training)
     */
    private Configuration(
            int lstmLayerSize,
            int miniBatchSize,
            int exampleLength,
            int tbpttLength,
            int numEpochs,
            int generateSamplesEveryNMinibatches,
            int nSamplesToGenerate,
            int nCharactersToSample,
            String generationInitialization,
            int randomSeed,
            String textSource,
            char[] validCharacters,
            String modelFileOutPath,
            String modelFileInPath
    ) {
        this.lstmLayerSize = lstmLayerSize;
        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;
        this.tbpttLength = tbpttLength;
        this.numEpochs = numEpochs;
        this.generateSamplesEveryNMinibatches = generateSamplesEveryNMinibatches;
        this.nSamplesToGenerate = nSamplesToGenerate;
        this.nCharactersToSample = nCharactersToSample;
        this.generationInitialization = generationInitialization;
        this.randomSeed = randomSeed;
        this.textSource = textSource;
        this.validCharacters = validCharacters;
        this.modelFileOutPath = modelFileOutPath;
        this.modelFileInPath = modelFileInPath;
    }

    public int getLstmLayerSize() {
        return lstmLayerSize;
    }

    public int getMiniBatchSize() {
        return miniBatchSize;
    }

    public int getExampleLength() {
        return exampleLength;
    }

    public int getTbpttLength() {
        return tbpttLength;
    }

    public int getNumEpochs() {
        return numEpochs;
    }

    public int getGenerateSamplesEveryNMinibatches() {
        return generateSamplesEveryNMinibatches;
    }

    public int getnSamplesToGenerate() {
        return nSamplesToGenerate;
    }

    public int getnCharactersToSample() {
        return nCharactersToSample;
    }

    public String getGenerationInitialization() {
        return generationInitialization;
    }

    public int getRandomSeed() {
        return randomSeed;
    }

    public String getTextSource() {
        return textSource;
    }

    public char[] getValidCharacters() {
        return validCharacters;
    }

    public String getModelFileOutPath() {
        return modelFileOutPath;
    }

    public String getModelFileInPath() {
        return modelFileInPath;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Initial settings are he original configuration with the Shakespeare Example
     */
    public static class Builder {
        private int lstmLayerSize = 200;
        private int miniBatchSize = 32;
        private int exampleLength = 1000;
        private int tbpttLength = 50;
        private int numEpochs = 1;
        private int generateSamplesEveryNMinibatches = 10;
        private int nSamplesToGenerate = 4;
        private int nCharactersToSample = 300;
        private String generationInitialization = null;
        private int randomSeed = 12345;
        private String textSource = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
        private char[] validCharacters = CharacterSets.getMinimalCharacterSet(); //Which characters are allowed? Others will be removed
        private String modelFileOutPath = "model.zip";
        private String modelFileInPath = null;

        public Builder setLstmLayerSize(int lstmLayerSize) {
            this.lstmLayerSize = lstmLayerSize;
            return this;
        }

        public Builder setMiniBatchSize(int miniBatchSize) {
            this.miniBatchSize = miniBatchSize;
            return this;
        }

        public Builder setExampleLength(int exampleLength) {
            this.exampleLength = exampleLength;
            return this;
        }

        public Builder setTbpttLength(int tbpttLength) {
            this.tbpttLength = tbpttLength;
            return this;
        }

        public Builder setNumEpochs(int numEpochs) {
            this.numEpochs = numEpochs;
            return this;
        }

        public Builder setGenerateSamplesEveryNMinibatches(int generateSamplesEveryNMinibatches) {
            this.generateSamplesEveryNMinibatches = generateSamplesEveryNMinibatches;
            return this;
        }

        public Builder setnSamplesToGenerate(int nSamplesToGenerate) {
            this.nSamplesToGenerate = nSamplesToGenerate;
            return this;
        }

        public Builder setnCharactersToSample(int nCharactersToSample) {
            this.nCharactersToSample = nCharactersToSample;
            return this;
        }

        public Builder setGenerationInitialization(String generationInitialization) {
            this.generationInitialization = generationInitialization;
            return this;
        }

        public Builder setRandomSeed(int randomSeed) {
            this.randomSeed = randomSeed;
            return this;
        }

        public Builder setTextSource(String textSource) {
            this.textSource = textSource;
            return this;
        }

        public Builder setValidCharacters(char[] validCharacters) {
            this.validCharacters = validCharacters;
            return this;
        }

        public Builder setModelFileOutPath(String modelFileOutPath) {
            this.modelFileOutPath = modelFileOutPath;
            return this;
        }

        public Builder setModelFileInPath(String modelFileInPath) {
            this.modelFileInPath = modelFileInPath;
            return this;
        }

        public Configuration build() {
            return new Configuration(
                    lstmLayerSize,
                    miniBatchSize,
                    exampleLength,
                    tbpttLength,
                    numEpochs,
                    generateSamplesEveryNMinibatches,
                    nSamplesToGenerate,
                    nCharactersToSample,
                    generationInitialization,
                    randomSeed,
                    textSource,
                    validCharacters,
                    modelFileOutPath,
                    modelFileInPath);
        }
    }
}
