from neat.species import DefaultSpeciesSet
from neat.math_util import mean, stdev
from neat.species import Species, GenomeDistanceCache
from six import iteritems, itervalues, iterkeys


class StrongSpeciesSet(DefaultSpeciesSet):

    def speciate(self, config, population, generation):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert isinstance(population, dict)

        compatibility_threshold = self.species_set_config.compatibility_threshold

        # Find the best representatives for each existing species.
        unspeciated = set(iterkeys(population))
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}
        new_members = {}
        for sid, s in iteritems(self.species):
            candidates = []
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            if len(candidates) > 0:
                ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
                new_rid = new_rep.key
                new_representatives[sid] = new_rid
                new_members[sid] = [new_rid]
                unspeciated.remove(new_rid)

        # Partition population into species based on genetic similarity.
        while unspeciated:
            gid = unspeciated.pop()
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in iteritems(new_representatives):
                rep = population[rid]
                d = distances(rep, g)
                if d < compatibility_threshold:
                    candidates.append((d, sid))

            if len(candidates) > 0:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in iteritems(new_representatives):
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        gdmean = mean(itervalues(distances.distances))
        gdstdev = stdev(itervalues(distances.distances))
        self.reporters.info(
            'Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev))
