# Von Tavel LSTM Char Modelling

Based on https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/character

Learning to speak [Bernese German](https://en.wikipedia.org/wiki/Bernese_German) using Recurrent Neural Networks with 
Long Short-Term Memory by looking at the [Rudolf von Tavel](https://en.wikipedia.org/wiki/Rudolf_von_Tavel)'s novels 
from http://gutenberg.spiegel.de/autor/rudolf-von-tavel-995


Some smaple output after the first few iterations
```
[...]
17:29:34.435 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener - Score at iteration 2598 is 66.54929165958953
17:29:35.319 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener - Score at iteration 2599 is 70.69355073889805
17:29:35.319 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - --------------------
17:29:35.319 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - Completed 130 minibatches of size 32x1000 characters
17:29:35.319 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - Sampling characters from network given initialization ""
17:29:36.177 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - ----- Sample 0 -----
17:29:36.177 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - &pfleche, i dahäre vo dei Oogen im Sattematt emel use het i Mei drücke. Aber er hü7chlet der Meinscht am Zode ghofftet hei, das sött ihm d’Sunne gsi, daß me sech brichte, wa’s i nes eissi Tage mit dem Fride Muscht gsi, daß si ne zwöit! Dä Brief isch es i müeß la läse.Vo der Lääre wieder d’Beine, wo-n
17:29:36.177 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - 
17:29:36.177 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - ----- Sample 1 -----
17:29:36.177 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - &blet, d’ Frou Oberwichtet. Dermeit isch es dem Ludig 191 worde, yne lieber gsi sygi, 's gä uf der Mittag uf das Schwyz ycho — kei Halbron bis züette. Er het gmeint gsi prächet werdi erscht mit de Händer d'Zügelschirmen ufe ghöre blaue – es isch vo der Chehier i ds Bett e chly meh gsi, daß nid wellen
17:29:36.177 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - 
17:29:36.177 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - ----- Sample 2 -----
17:29:36.177 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - &yßen ele, schwär i der lassenti ja überen und Manuerloß» – Chumpfer!97 Albei, so stüber me mit ds hopier gförchtete Morge d’Ändwulk teit. Vo der Allee jitz viel mir öppis agsetzt. Me het sech gäge Ryter us em Schlacht wüssi, so isch ese Schtrlus gsi, wo es nume meh het, wo me nie nume nid sieß grave
17:29:36.177 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - 
17:29:36.177 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - ----- Sample 3 -----
17:29:36.177 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - & emekeret, bi Arte, wo nid cho, und  er het ne-n-anto usfahrhet yrüte; aber ohni mit syne Röffismette Lüt me gchressiger, sobald ds o des Huus keir »Aha? — Mir mängem Dascht di gwohnesige Räselete gchrisset: »I ha es Möntsch gheisse. Si müeßti für Eue.Über neu gange scho lieb hiehet der Jukter Gabrz
17:29:36.177 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - 
17:29:42.040 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener - Score at iteration 2600 is 78.3843936511528
17:29:42.821 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener - Score at iteration 2601 is 74.05114168655417
[...]
```

Some sample output after not too many iterations more
```
[...]
19:47:54.770 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener - Score at iteration 9598 is 61.833213419347814
19:47:55.708 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener - Score at iteration 9599 is 62.13872776840547
19:47:55.708 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - --------------------
19:47:55.708 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - Completed 480 minibatches of size 32x1000 characters
19:47:55.708 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - Sampling characters from network given initialization ""
19:47:56.703 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - ----- Sample 0 -----
19:47:56.703 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - Üsserobe g’antwortet het: »I trieb i schludere? Und das Hand jitz di Herre!»So het ds Wasser und Roß usegtue, bevor der Herr Lombach uf d'Sunnen umacht und uf d'Schwyzer gar nüt ghörti derfür gange, bis der Großs sölli uffahre, daß der Herr Manuel afah dry chunnt.Es isch verno, und wyter sy de öppen 
19:47:56.703 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - 
19:47:56.703 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - ----- Sample 1 -----
19:47:56.703 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - Überlärme z’trotz isch. Es het na däm Heiwäg drinns für Blueme z'fähle. Er isch usgseh sy. Aber dä Bueb het i der Jumpfer Müntigschtimmer ds Meitli läbhafts gseit und het nachhär nüt. Jitz wird er sech scho öppis ganz flyßig als Ärnt ga z’ gseh, wie wenn er wieder vor sys Gsicht. Aber nachem!.., was 
19:47:56.703 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - 
19:47:56.703 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - ----- Sample 2 -----
19:47:56.703 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - Überluet zwöim Griengi liecht umenandere gschtoße het, vowäge ds Härz hoffe-n-i am ganze übermüetig und het uf der Meinung ganz hällem Gsicht schtatt abz iereh worde. Vo sy Schyffeschyn het me scho mängisch zsäme gstige, heiwäche Müntschi gmahnet, daß er synerzyt wäri, so chönne hütt no rächt gsi — s
19:47:56.703 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - 
19:47:56.703 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - ----- Sample 3 -----
19:47:56.703 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - Überzügt hätti o druifschter und um nes, gtörchlet und under Dacht über ds grobem Schlaf yne, und ds Jetti, wo scho läbe schlüüri und het usgschtoß und zum Schloß und Warde-so ha-n-i kolfer Arbeit gschtellt z’sy, het si i ihrem Predigärt-1-Kardanag gseit, i well nume mit z'rächtem Liecht. Du du chuet
19:47:56.703 [main] INFO ch.be.von.tavel.LSTMCharModellingExample - 
19:47:57.731 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener - Score at iteration 9600 is 68.82843644664992
19:47:58.787 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener - Score at iteration 9601 is 64.34566548607292
[...]
```

Some useful links about RNNs and LSTM:
- [https://arxiv.org/abs/1506.00019]( A Critical Review of Recurrent Neural Networks for Sequence Learning) by  Zachary C. Lipton, John Berkowitz, Charles Elkan
- [http://karpathy.github.io/2015/05/21/rnn-effectiveness/](The Unreasonable Effectiveness of Recurrent Neural Networks) by Andrej Karpathy