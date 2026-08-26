#!/usr/bin/env python3
"""The terrain work -- terrain scenes, built rather than assumed.

Sections 1 to 7 ran entirely on `type="plane"`, an infinite frictionless-free
flat surface. 8.1 argues that is a lie, and this file is what makes the
argument checkable: it generates REAL geometry the robot can walk onto, trip
over and fall off.

Three families, all parameterised, all writing a scene MuJoCo actually loads:

  ramp(deg)          a long inclined slab, approached from flat ground
  steps(rise, n)     a run of square steps up
  gap(width)         flat ground with a hole in it

Everything else in the terrain work imports from here, so a terrain fix lands in one
place. Nothing here reads a measurement; the analysis scripts do that.
"""
import os
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
SCENE = ROOT / "mujoco/resources/robots/h1_2/scene_full.xml"
OUT = ROOT / "mujoco/resources/robots/h1_2"
BASE = SCENE.read_text()

# Where the flat run-up ends and the terrain begins. The robot spawns at x=0
# and the policy needs a few gait cycles to reach steady state, so terrain that
# starts at x=0 measures the spawn transient instead of the terrain.
X0 = 2.0


# Terrain gets its own MATTE material. The first renders showed a band of
# moire across every slab, which looked like z-fighting and was not: the floor
# carries reflectance="0.12" and a 14 m slab mirrors its checker texture. A
# matte material with specular 0 removes it, and the geometry never changed.
TERRAIN_MAT = ('    <material name="terra" rgba="0.34 0.37 0.44 1"\n'
               '              reflectance="0" specular="0.05" shininess="0.05"/>\n')


def _write(tag, extra, keep_floor=True):
    """Splice geometry into scene_full.xml and load it.

    The floor STAYS by default: the robot has to walk from flat ground onto
    the terrain, because a robot that spawns already on a slope has no
    baseline to be compared against.
    """
    s = BASE
    if not keep_floor:
        # RENAMING the floor does not remove it. The first version replaced
        # name="floor" with name="_unused" and left an infinite, fully
        # collidable plane in the world: the robot walked straight across a
        # 1.5 m hole with its feet at z=0.04 INSIDE the gap, on nothing, and
        # every gap width from 100 mm to 1500 mm reported "crossed". The
        # selftest passed too, because it only checked that no geom was NAMED
        # "floor". Delete the element.
        i = s.index('<geom name="floor" type="plane"')
        j = s.index('/>', i) + 2
        s = s[:i] + s[j:]
    s = s.replace("  </asset>", TERRAIN_MAT + "  </asset>", 1)
    s = s.replace("  </worldbody>", extra + "  </worldbody>", 1)
    p = OUT / ("_terrain_%s.xml" % tag)
    p.write_text(s)
    m = mujoco.MjModel.from_xml_path(str(p))
    p.unlink()
    return m


# The ramp has to be WIDE. This policy veers: on flat ground it wanders to
# |y| = 4.2 m over 25 s while walking 10 m forward. A 3 m wide ramp measured
# how long the robot took to walk off the SIDE of it, and reported that as a
# slope failure at 2 and 4 degrees while 10 degrees "survived". A terrain
# parameter that is narrower than the robot's own lateral drift is not
# measuring terrain.
def ramp(deg, length=6.0, width=14.0, tag="ramp", lift=0.0):
    """An inclined slab starting at X0, rising at `deg`.

    Built as a rotated box rather than a hfield: a box has exact contact
    geometry and one friction value, so a slip result is about the SLOPE and
    not about how finely a heightfield was sampled.

    The slab is SUNK by 3 mm so its near edge sits just under the floor plane
    rather than exactly on it. Coplanar faces z-fight, and the render showed a
    band of moire along the whole join that reads as a broken mesh. Sinking
    rather than lifting keeps the walking surface continuous: a 3 mm lip is
    below this robot's foot clearance and does not change the contact, whereas
    lifting would put a step in front of the gradient and measure the step.

    `lift` is kept for explicit override and defaults to that sink.
    """
    th = np.radians(deg)
    # the slab's centre, so its upper face starts at (X0, 0) on the ground
    half = length / 2.0
    cx = X0 + half * np.cos(th)
    cz = half * np.sin(th) - 0.05 * np.cos(th) + lift - 0.003
    g = ('    <body name="ramp" pos="%.4f 0 %.4f" euler="0 %.6f 0">\n'
         '      <geom name="ramp_geom" type="box" size="%.4f %.4f 0.05"\n'
         '            material="terra" condim="3"\n'
         '            friction="1.0 0.05 0.05"/>\n'
         '    </body>\n' % (cx, cz, -th, half, width / 2.0))
    return _write(tag, g)


def steps(rise, n=6, tread=0.30, width=14.0, tag="steps"):
    """`n` square steps of `rise` height and `tread` depth, starting at X0."""
    g = ""
    for i in range(n):
        x = X0 + tread * i + tread / 2.0
        z = rise * (i + 1) / 2.0
        g += ('    <body name="step%d" pos="%.4f 0 %.4f">\n'
              '      <geom name="step%d_geom" type="box" size="%.4f %.4f %.4f"\n'
              '            material="terra" condim="3"\n'
              '            friction="1.0 0.05 0.05"/>\n'
              '    </body>\n'
              % (i, x, z, i, tread / 2.0, width / 2.0, rise * (i + 1) / 2.0))
    return _write(tag, g)


# How deep the hole is. Not infinite: with no floor under the gap a robot
# that falls in keeps falling, and the 400 mm take ended at z = -137 m, which
# is physically correct and unwatchable. 1.2 m is deeper than the robot is
# tall, so falling in is unambiguously a failure, and the fall terminates.
GAP_DEPTH = 1.2


def gap(width_m, tag="gap"):
    """Flat ground with a hole. The plane floor is REMOVED and replaced by two
    slabs, because you cannot cut a hole in an infinite plane."""
    g = ('    <body name="pit" pos="%.4f 0 %.4f">\n'
         '      <geom name="pit_geom" type="box" size="%.4f 7.0 0.05"\n'
         '            material="terra" condim="3"\n'
         '            friction="1.0 0.05 0.05"/>\n'
         '    </body>\n' % (X0 + width_m / 2.0, -GAP_DEPTH, width_m / 2.0 + 2.5))
    for i, (x0, x1) in enumerate([(-4.0, X0), (X0 + width_m, X0 + width_m + 26.0)]):
        cx = (x0 + x1) / 2.0
        g += ('    <body name="ground%d" pos="%.4f 0 -0.05">\n'
              '      <geom name="ground%d_geom" type="box" size="%.4f 7.0 0.05"\n'
              '            material="terra" condim="3"\n'
              '            friction="1.0 0.05 0.05"/>\n'
              '    </body>\n' % (i, cx, i, (x1 - x0) / 2.0))
    return _write(tag, g, keep_floor=False)


def flat(tag="flat"):
    """The baseline: scene_full.xml unchanged. Having it here means every
    comparison in the section runs through the same loader."""
    return _write(tag, "")


def selftest():
    """Assert the geometry is what it claims, because a scene that LOADS is not
    the same as a scene that is correct. A ramp whose slab is buried in the
    floor still loads, still simulates, and quietly measures flat ground.
    """
    import mujoco as mj
    # a ramp's far end must be at length*sin(deg) above the ground
    for deg in (5.0, 10.0, 20.0):
        m = ramp(deg, length=6.0)
        d = mj.MjData(m)
        mj.mj_forward(m, d)
        gid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "ramp_geom")
        bid = m.geom_bodyid[gid]
        # top face of the far end, in world
        R = d.xmat[bid].reshape(3, 3)
        c = d.xpos[bid]
        far = c + R @ np.array([3.0, 0.0, 0.05])
        want = 6.0 * np.sin(np.radians(deg))
        assert abs(far[2] - want) < 0.02, \
            "ramp %g deg: far end at %.4f m, expected %.4f" % (deg, far[2], want)
        near = c + R @ np.array([-3.0, 0.0, 0.05])
        assert abs(near[2]) < 0.02, \
            "ramp %g deg: near end at %.4f m, should meet the ground" % (deg, near[2])
    # steps: the top of step i must be exactly rise*(i+1)
    m = steps(0.10, n=4)
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    for i in range(4):
        gid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "step%d_geom" % i)
        top = d.geom_xpos[gid][2] + m.geom_size[gid][2]
        assert abs(top - 0.10 * (i + 1)) < 1e-6, \
            "step %d top at %.4f, expected %.4f" % (i, top, 0.10 * (i + 1))
    # a gap must have NO geometry over the hole
    m = gap(0.40)
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
    assert "floor" not in names, "gap scene still has the infinite plane"
    # The far slab must outlast the run. At 0.9 m/s a 20 s take covers about
    # 14 m, and the original 8 m slab ended at x=10.6: every gap result was a
    # robot that had walked off the END of the world, reported as "crossed".
    planes = [i for i in range(m.ngeom)
              if m.geom_type[i] == mj.mjtGeom.mjGEOM_PLANE
              and m.geom_contype[i]]
    assert not planes, (
        "gap scene still has %d collidable plane(s): renaming the floor does "
        "not remove it, and the robot will walk across the hole on it"
        % len(planes))
    gid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "ground1_geom")
    far = d.geom_xpos[gid][0] + m.geom_size[gid][0]
    assert far >= 25.0, "far slab ends at x=%.1f, too short for a 20 s take" % far
    # the pit floor has to be well below the robot, or it becomes a shallow
    # step the robot simply walks over and the gap stops being a gap
    pid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "pit_geom")
    pit_top = d.geom_xpos[pid][2] + m.geom_size[pid][2]
    assert pit_top < -0.9, "pit floor at z=%.2f is not a hole, it is a step" % pit_top
    print("  terrain selftest OK: ramps meet the ground and rise correctly, "
          "step tops exact, gap has no floor plane")


if __name__ == "__main__":
    print("--- the scenes the terrain work runs on ---")
    selftest()
    print()
    for name, m in (("flat", flat()), ("ramp 10 deg", ramp(10.0)),
                    ("steps 0.10 x6", steps(0.10)), ("gap 0.40 m", gap(0.40))):
        print("  %-16s ngeom %3d   nbody %3d" % (name, m.ngeom, m.nbody))
