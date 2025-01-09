// ------------------------------------------------------------------------------- //
// Advanced Kalman Filtering and Sensor Fusion Course - Unscented Kalman Filter
//
// ####### STUDENT FILE #######
//
// Usage:
// -Rename this file to "kalmanfilter.cpp" if you want to use this code.

#include "kalmanfilter.h"
#include "utils.h"

// -------------------------------------------------- //
// YOU CAN USE AND MODIFY THESE CONSTANTS HERE
constexpr double ACCEL_STD = 0.05;
constexpr double GYRO_STD = 0.01/180.0 * M_PI;
constexpr double INIT_VEL_STD = 2;
constexpr double INIT_PSI_STD = 5.0/180.0 * M_PI;
constexpr double GPS_POS_STD = 3.0;
constexpr double LIDAR_RANGE_STD = 3.0;
constexpr double LIDAR_THETA_STD = 0.02;
// -------------------------------------------------- //

// ----------------------------------------------------------------------- //
// USEFUL HELPER FUNCTIONS
VectorXd normaliseState(VectorXd state)
{
    state(2) = wrapAngle(state(2));
    return state;
}
VectorXd normaliseLidarMeasurement(VectorXd meas)
{
    meas(1) = wrapAngle(meas(1));
    return meas;
}
std::vector<VectorXd> generateSigmaPoints(VectorXd state, MatrixXd cov)
{
    std::vector<VectorXd> sigmaPoints {state};

    int n = state.size();
    sigmaPoints.reserve(n*2+1);

    int k = 3 - n;
    auto lmbda = sqrt(n+k);
    MatrixXd sP = cov.llt().matrixL();

    for (int i = 0; i < n; i++) {
        sigmaPoints.emplace_back(state + lmbda * sP.col(i));
        sigmaPoints.emplace_back(state - lmbda * sP.col(i));
    }

    return sigmaPoints;
}

std::vector<double> generateSigmaWeights(unsigned int n)
{
    double k = 3.0 - n;
    auto div = n+k;
    std::vector<double> weights { {k/div} };
    for (unsigned i = 0; i < 2*n; i++) {
        weights.emplace_back(0.5/div);
    } 

    return weights;
}

VectorXd lidarMeasurementModelSimle(VectorXd state, double beaconX, double beaconY) {
    VectorXd zhat = VectorXd::Zero(2);
    double px = state[0];
    double py = state[1];
    double pPhi = state[2];
    double pV = state[3];

    double dx = px-beaconX;
    double dy = py-beaconY;

    zhat(0) = sqrt(dx*dx + dy*dy);
    zhat(1) = atan2(dy, dx) - pPhi;

    return zhat;
}

VectorXd lidarMeasurementModel(VectorXd aug_state, double beaconX, double beaconY)
{
    VectorXd z_hat = VectorXd::Zero(2);

    double px = aug_state[0];
    double py = aug_state[1];
    double pPhi = aug_state[2];
    double vR = aug_state[4];
    double vPhi = aug_state[5];

    double dx = px-beaconX;
    double dy = py-beaconY;

    z_hat(0) = sqrt(dx*dx + dy*dy) + vR;
    z_hat(1) = atan2(dy, dx) - pPhi + vPhi;

    return z_hat;
}

VectorXd vehicleProcessModel(VectorXd aug_state, double psi_dot, double dt)
{
    VectorXd new_state = VectorXd::Zero(4);
    double x = aug_state(0);
    double y = aug_state(1);
    double psi = aug_state(2);
    double v = aug_state(3);
    double psi_nosie = aug_state(4);
    double accel_noise = aug_state(5);

    new_state(0) = x + dt * v*cos(psi);
    new_state(1) = y + dt * v*sin(psi);
    new_state(2) = psi + dt * (psi_dot + psi_nosie);
    new_state(3) = v + dt * accel_noise;
    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE

    // ----------------------------------------------------------------------- //

    return new_state;
}
// ----------------------------------------------------------------------- //

void KalmanFilter::handleLidarMeasurement(LidarMeasurement meas, const BeaconMap& map)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Update Step for the Lidar Measurements in the 
        // section below.
        // HINT: Use the normaliseState() and normaliseLidarMeasurement() functions
        // to always keep angle values within correct range.
        // HINT: Do not normalise during sigma point calculation!
        // HINT: You can use the constants: LIDAR_RANGE_STD, LIDAR_THETA_STD
        // HINT: The mapped-matched beacon position can be accessed by the variables
        // map_beacon.x and map_beacon.y
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE

        BeaconData map_beacon = map.getBeaconWithId(meas.id); // Match Beacon with built in Data Association Id
        if (meas.id != -1 && map_beacon.id != -1) // Check that we have a valid beacon match
        {
            VectorXd zCur = VectorXd::Zero(2);
            zCur(0) = meas.range;
            zCur(1) = meas.theta;

            VectorXd aState = VectorXd::Zero(6);
            for (int i = 0; i < 4; i++) {
              aState(i) = state(i);
            }
            MatrixXd aCov = MatrixXd::Zero(6, 6);
            for (int r = 0; r < 4; r++) {
              for (int c = 0; c < 4; c++) {
                aCov(r, c) = cov(r, c);
              }
            }
            aCov(4, 4) = LIDAR_RANGE_STD * LIDAR_RANGE_STD;
            aCov(5, 5) = LIDAR_THETA_STD * LIDAR_THETA_STD;

            auto sigmas = generateSigmaPoints(aState, aCov);
            auto weights = generateSigmaWeights(6);


            std::vector<VectorXd> zis;
            for (size_t i = 0; i < sigmas.size(); i++) {
                zis.emplace_back(lidarMeasurementModel(sigmas[i], map_beacon.x, map_beacon.y));
            }

            VectorXd zHat = VectorXd::Zero(2);
            for (size_t i = 0; i < sigmas.size(); i++) {
                zHat += weights[i] * zis[i];
            }

            MatrixXd S = MatrixXd::Zero(2, 2);
            MatrixXd Pxz = MatrixXd::Zero(4, 2);
            for (size_t i = 0; i < sigmas.size(); i++) {
                auto zDiff = normaliseLidarMeasurement(zis[i] - zHat);
                auto xDiff = normaliseState(sigmas[i].head(4) - state);
                S += weights[i] * zDiff * zDiff.transpose();
                Pxz += weights[i] * xDiff * zDiff.transpose();
            }

            MatrixXd K = Pxz * S.inverse();
            VectorXd vHat = normaliseLidarMeasurement(zCur - zHat);
            state = state + K * vHat;
            cov = cov - K * S * K.transpose();
            
        }
        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    }
}

void KalmanFilter::predictionStep(GyroMeasurement gyro, double dt)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        int n = state.size();
        int an = n+2;
        VectorXd aState = VectorXd::Zero(an);
        for (int i = 0; i < n; i++) {
            aState(i) = state(i);
        }

        // TODO: try to comment.
        // aState(4) = GYRO_STD;
        // aState(5) = ACCEL_STD;

        MatrixXd aCov = MatrixXd::Zero(an,an);
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                aCov(r, c) = cov(r, c);
            }
        }
        aCov(n, n) = GYRO_STD*GYRO_STD;
        aCov(n+1, n+1) = ACCEL_STD*ACCEL_STD;

        auto sigmas = generateSigmaPoints(aState, aCov);
        auto weights = generateSigmaWeights(an);

        std::vector<VectorXd> pred;
        for (VectorXd paug : sigmas) {
            pred.emplace_back(vehicleProcessModel(paug, gyro.psi_dot, dt));
        }

        VectorXd resState = VectorXd::Zero(4);
        for (int i = 0; i < pred.size(); i++) {
            resState += weights[i] * pred[i];
        }
        resState = normaliseState(resState);

        MatrixXd resCov = MatrixXd::Zero(4,4);
        for (int i = 0; i < pred.size(); i++) {
            VectorXd diff = normaliseState(resState-pred[i]);
            resCov += weights[i] * diff * diff.transpose();
        }
        
        // Implement The Kalman Filter Prediction Step for the system in the  
        // section below.
        // HINT: Assume the state vector has the form [PX, PY, PSI, V].
        // HINT: Use the Gyroscope measurement as an input into the prediction step.
        // HINT: You can use the constants: ACCEL_STD, GYRO_STD
        // HINT: Use the normaliseState() function to always keep angle values within correct range.
        // HINT: Do NOT normalise during sigma point calculation!
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE


        // ----------------------------------------------------------------------- //

        setState(resState);
        setCovariance(resCov);
    } 
}

void KalmanFilter::handleGPSMeasurement(GPSMeasurement meas)
{
    // All this code is the same as the LKF as the measurement model is linear
    // so the UKF update state would just produce the same result.
    if(isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        VectorXd z = Vector2d::Zero();
        MatrixXd H = MatrixXd(2,4);
        MatrixXd R = Matrix2d::Zero();

        z << meas.x,meas.y;
        H << 1,0,0,0,0,1,0,0;
        R(0,0) = GPS_POS_STD*GPS_POS_STD;
        R(1,1) = GPS_POS_STD*GPS_POS_STD;

        VectorXd z_hat = H * state;
        VectorXd y = z - z_hat;
        MatrixXd S = H * cov * H.transpose() + R;
        MatrixXd K = cov*H.transpose()*S.inverse();

        state = state + K*y;
        cov = (MatrixXd::Identity(4,4) - K*H) * cov;

        setState(state);
        setCovariance(cov);
    }
    else
    {
        // You may modify this initialisation routine if you can think of a more
        // robust and accuracy way of initialising the filter.
        // ----------------------------------------------------------------------- //
        // YOU ARE FREE TO MODIFY THE FOLLOWING CODE HERE

        VectorXd state = Vector4d::Zero();
        MatrixXd cov = Matrix4d::Zero();

        state(0) = meas.x;
        state(1) = meas.y;
        cov(0,0) = GPS_POS_STD*GPS_POS_STD;
        cov(1,1) = GPS_POS_STD*GPS_POS_STD;
        cov(2,2) = INIT_PSI_STD*INIT_PSI_STD;
        cov(3,3) = INIT_VEL_STD*INIT_VEL_STD;

        setState(state);
        setCovariance(cov);

        // ----------------------------------------------------------------------- //
    }             
}

void KalmanFilter::handleLidarMeasurements(const std::vector<LidarMeasurement>& dataset, const BeaconMap& map)
{
    // Assume No Correlation between the Measurements and Update Sequentially
    for(const auto& meas : dataset) {handleLidarMeasurement(meas, map);}
}

Matrix2d KalmanFilter::getVehicleStatePositionCovariance()
{
    Matrix2d pos_cov = Matrix2d::Zero();
    MatrixXd cov = getCovariance();
    if (isInitialised() && cov.size() != 0){pos_cov << cov(0,0), cov(0,1), cov(1,0), cov(1,1);}
    return pos_cov;
}

VehicleState KalmanFilter::getVehicleState()
{
    if (isInitialised())
    {
        VectorXd state = getState(); // STATE VECTOR [X,Y,PSI,V,...]
        return VehicleState(state[0],state[1],state[2],state[3]);
    }
    return VehicleState();
}

void KalmanFilter::predictionStep(double dt){}
