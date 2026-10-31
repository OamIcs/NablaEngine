"""What the engine needs of a MuJoCo model.

  nablaengine.model.fields      every MjModel field of mujoco 3.17 classified (consumed / checked / default / derived /
                                irrelevant): the completeness proof of the checker
  nablaengine.model.capability  the input validation: findings with codes, UnsupportedModel, require() (called by
                                nablaengine.jax_ffi.init / check_model, EnginePhysics, nablaengine.fixtures.load_model
                                and g1_engine.kinematics.check_model)
"""
