"""Single-interval update formulas, independent of model loading and I/O."""


def euler_step(state, dt, velocity):
    return state + dt * velocity, velocity


def ab2_step(state, dt, velocity, previous_velocity):
    combined = 1.5 * velocity - 0.5 * previous_velocity
    return state + dt * combined, combined


def sg_step(state, dt, velocity, shifted_velocity, scale):
    combined = velocity + scale * (velocity - shifted_velocity)
    return state + dt * combined, combined


def heun_step(state, dt, velocity, predictor_velocity):
    combined = 0.5 * (velocity + predictor_velocity)
    return state + dt * combined, combined


def cab2_step(state, time, next_time, grid, index, derivative, history, theta):
    """Variable-step CAB in y=x/t, lambda=(1-t)/t; history stores clean-v."""
    lam = (1 - grid[index - 2 : index + 2]) / grid[index - 2 : index + 2]
    previous_previous_step, previous_step, step = lam[1:] - lam[:-1]
    ratio = step / previous_step
    previous_ratio = previous_step / previous_previous_step
    previous_derivative, previous_previous_derivative = history[-1], history[-2]

    extrapolated = (
        1 + previous_ratio
    ) * previous_derivative - previous_ratio * previous_previous_derivative
    predictor = state / time + step * (
        (1 + ratio / 2) * derivative - ratio / 2 * previous_derivative
    )
    next_state = next_time * (predictor + theta * step * (derivative - extrapolated))
    effective_velocity = (next_state - state) / (next_time - time)
    details = dict(
        k=step,
        r=ratio,
        r_prev=previous_ratio,
        predictor=predictor,
        extrapolated=extrapolated,
        theta=theta,
    )
    return next_state, effective_velocity, details
