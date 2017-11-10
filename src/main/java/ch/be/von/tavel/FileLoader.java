package ch.be.von.tavel;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Random;
import java.util.UUID;

public class FileLoader {

    private static final Logger LOG = LoggerFactory.getLogger(FileLoader.class);

    private final int miniBatchSize;
    private final int sequenceLength;
    private final int rng;
    private final char[] validCharacters;

    /**
     * Auxiliary class for opening training data or downloading training data and storing it locally (temp directory).
     * Then set up and return a simple DataSetIterator that does vectorization based on the text.
     *  @param miniBatchSize  Number of text segments in each training mini-batch
     * @param sequenceLength Number of characters in each text segment.
     * @param rng
     * @param validCharacters
     */
    public FileLoader(int miniBatchSize, int sequenceLength, int rng, char[] validCharacters) {
        this.miniBatchSize = miniBatchSize;
        this.sequenceLength = sequenceLength;
        this.rng = rng;
        this.validCharacters = validCharacters;
    }

    public CharacterIterator getIterator(String location) throws IOException {
        if (location.startsWith("http")) {
            return getIteratorFromWeb(location);
        } else {
            return getIteratorFromFileLocation(location);
        }
    }

    private CharacterIterator getIteratorFromWeb(String url) throws IOException {
        String tempDir = System.getProperty("java.io.tmpdir");
        String fileLocation = tempDir + "/LSTMCharModellingExample-" + UUID.randomUUID().toString() + ".txt"; //Storage location from downloaded file
        File f = new File(fileLocation);
        LOG.info("Downloading {} to {}", url, f.getAbsolutePath());
        FileUtils.copyURLToFile(new URL(url), f);
        return getIteratorFromFile(f);
    }

    private CharacterIterator getIteratorFromFileLocation(String fileLocation) throws IOException {
        File f = new File(fileLocation);
        return getIteratorFromFile(f);
    }

    private CharacterIterator getIteratorFromFile(File f) throws IOException {
        LOG.info("Opening {}", f.getAbsolutePath());

        if (!f.exists()) throw new IOException("File does not exist: " + f.getAbsolutePath());

        return new CharacterIterator(f.getAbsolutePath(), Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, validCharacters, new Random(rng));
    }
}
