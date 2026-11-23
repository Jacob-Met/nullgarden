# Synthetic operator-recovery controls

Run `node src/operator-course.mjs > operator-course.json` to materialize the fixed nondimensional oscillator and periodic-field courses. The emitter needs Node 22 or newer and no solver checkout. It provides exact instantaneous right-hand sides and independent midpoint-integrated state paths using 32 substeps per observation interval.

The oscillator acceleration is command times cosine of position. The periodic field uses centered advection and diffusion on a uniform grid. Challenge courses change initial state and excitation; the field also changes from 12 to 15 sites. Within each original run, the sham advances response labels seven time steps at the same site while preserving every input. The derivative and state tolerances are in the emitted settings.

Three out-of-grammar controls repeat an observed state with different required derivatives. The delay response depends on an unobserved previous value. The Lorenz acceleration depends on the hidden third coordinate. The periodic third difference depends on a second neighbour outside the supplied stencil. Their exact same-input half-ranges are 1, 30 and 1. These numbers bound the worst residual of every deterministic model of the recorded inputs on the respective finite examples.
