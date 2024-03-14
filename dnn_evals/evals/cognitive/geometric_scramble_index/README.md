# GeometricScrambleIndex

- Geom(Invariance)Scramble(Sensitivity)Index(GISSI; or GSI): the degree of tolerance to geometric transforms minus the degree of tolerance to scrambling. In practice, the index varies from zero (equally tolerant to geometric transforms and scrambling) to one (perfectly tolerant to geometric transforms and zero-tolerance for scrambling). Negative values are possible (indicating greater tolerance to srambling than geometric transforms), but unlikely.

- InstanceClassification (KNN): compute the similarity between an item (intact and scrambled variants) and all geom transformed copies of items, see how often the instance is correclty classified as the correct instance (similarity-weighted voting from 10NN instance labels). A model with geometric, instance-specific representations should be able to identify the intact but not scrambled versions.

- CategoryClassification (KNN): compute the similarity between an item (intact and scrambled variants) and all geom transformed copies of items, see how often the instance is correclty classified as the correct category (similarity-weighted voting from 10NN category labels). A model with geometric, category-specific representations should be able to identify the intact but not scrambled versions.

## Scramble ideas 
- add a "grid mask" control where we overlay grid lines to see if we are underestimating the scramble sensitivity (tolerance might be higher, but we are introducing grid-lines and that could be falsely distinguishing intact and scrambled versions)
- blocks at different resolutions with/without random rotation of blocks (rotation set could be 90,270 or 90,180,270
- use an autoencoder to "erase" the seams between blocks
- Erez' diffeomorphic scramble
- https://dangeng.github.io/visual_anagrams/