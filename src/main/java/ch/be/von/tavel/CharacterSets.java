package ch.be.von.tavel;

import java.util.LinkedList;
import java.util.List;

public class CharacterSets {

    /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
    public static char[] getMinimalCharacterSet(){
        List<Character> validChars = new LinkedList<>();
        for(char c='a'; c<='z'; c++) validChars.add(c);
        for(char c='A'; c<='Z'; c++) validChars.add(c);
        for(char c='0'; c<='9'; c++) validChars.add(c);
        char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
        for( char c : temp ) validChars.add(c);
        char[] out = new char[validChars.size()];
        int i=0;
        for( Character c : validChars ) out[i++] = c;
        return out;
    }

    /** As per getMinimalCharacterSet(), but with a few extra characters */
    public static char[] getDefaultCharacterSet(){
        List<Character> validChars = new LinkedList<>();
        for(char c : getMinimalCharacterSet() ) validChars.add(c);
        char[] additionalChars = {'@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_',
                '\\', '|', '<', '>'};
        for( char c : additionalChars ) validChars.add(c);
        char[] out = new char[validChars.size()];
        int i=0;
        for( Character c : validChars ) out[i++] = c;
        return out;
    }
}
