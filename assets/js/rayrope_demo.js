// ============== RayRoPE Interactive Demo (JavaScript version) ==============
// This module provides the attention visualization demo for RayRoPE

(function(global) {
    'use strict';

    // Constants
    const MAX_DEPTH = 100.0;

    // Fixed parameters
    const HEAD_DIM = 12;
    const PATCHES_X = 9;
    const PATCHES_Y = 9;
    const NUM_PATCHES = PATCHES_X * PATCHES_Y;
    const IMAGE_WIDTH = 9;
    const IMAGE_HEIGHT = 9;
    const NUM_RAYS_PER_PATCH = 1;
    const FREQ_BASE = 3.0;

    // Position encoding config: 0_3d + d_pj
    const USE_P0 = true;
    const USE_PD = true;
    const DENC_TYPE = 'inv_d';

    // Compute rope dimensions
    const ROPE_COORD_DIM = 3 * (USE_P0 ? 1 : 0) + NUM_RAYS_PER_PATCH * 3 * (USE_PD ? 1 : 0);
    const NUM_ROPE_FREQS = Math.floor(HEAD_DIM / (2 * ROPE_COORD_DIM));

    // Track if 3D plot has been initialized (to preserve camera view)
    let plot3DInitialized = false;

    // ============== Matrix utilities ==============

    function zeros(shape) {
        if (shape.length === 1) return new Array(shape[0]).fill(0);
        return new Array(shape[0]).fill(null).map(() => zeros(shape.slice(1)));
    }

    function eye4() {
        return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
    }

    function matmul4x4(A, B) {
        const C = zeros([4, 4]);
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                for (let k = 0; k < 4; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    function matvec4(M, v) {
        const r = [0, 0, 0, 0];
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                r[i] += M[i][j] * v[j];
            }
        }
        return r;
    }

    function invertSE3(w2c) {
        const Rinv = [
            [w2c[0][0], w2c[1][0], w2c[2][0]],
            [w2c[0][1], w2c[1][1], w2c[2][1]],
            [w2c[0][2], w2c[1][2], w2c[2][2]]
        ];
        const t = [w2c[0][3], w2c[1][3], w2c[2][3]];
        const tNew = [
            -(Rinv[0][0]*t[0] + Rinv[0][1]*t[1] + Rinv[0][2]*t[2]),
            -(Rinv[1][0]*t[0] + Rinv[1][1]*t[1] + Rinv[1][2]*t[2]),
            -(Rinv[2][0]*t[0] + Rinv[2][1]*t[1] + Rinv[2][2]*t[2])
        ];
        return [
            [Rinv[0][0], Rinv[0][1], Rinv[0][2], tNew[0]],
            [Rinv[1][0], Rinv[1][1], Rinv[1][2], tNew[1]],
            [Rinv[2][0], Rinv[2][1], Rinv[2][2], tNew[2]],
            [0, 0, 0, 1]
        ];
    }

    function liftK(K3x3) {
        return [
            [K3x3[0][0], K3x3[0][1], K3x3[0][2], 0],
            [K3x3[1][0], K3x3[1][1], K3x3[1][2], 0],
            [K3x3[2][0], K3x3[2][1], K3x3[2][2], 0],
            [0, 0, 0, 1]
        ];
    }

    function invertK(K) {
        const out = [[0,0,0],[0,0,0],[0,0,0]];
        out[0][0] = 1.0 / (K[0][0] + 1e-9);
        out[1][1] = 1.0 / (K[1][1] + 1e-9);
        out[0][2] = -K[0][2] / (K[0][0] + 1e-9);
        out[1][2] = -K[1][2] / (K[1][1] + 1e-9);
        out[2][2] = 1.0;
        return out;
    }

    function normalizeK(K, imgW, imgH) {
        return [
            [K[0][0] / imgW, K[0][1], K[0][2] / imgW - 0.5],
            [K[1][0], K[1][1] / imgH, K[1][2] / imgH - 0.5],
            [K[2][0], K[2][1], K[2][2]]
        ];
    }

    function norm3(v) {
        return Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    }

    function createRotationMatrixY(angleDeg) {
        const rad = angleDeg * Math.PI / 180;
        const c = Math.cos(rad), s = Math.sin(rad);
        return [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ];
    }

    function createW2C(tx, tz, rotY) {
        const R = createRotationMatrixY(rotY);
        const rad = rotY * Math.PI / 180;
        // Camera at world position (tx, 0, tz), transformed to camera space
        const tCam = [
            -tx * Math.cos(rad) - tz * Math.sin(rad),
            0,
            tx * Math.sin(rad) - tz * Math.cos(rad)
        ];
        return [
            [R[0][0], R[0][1], R[0][2], tCam[0]],
            [R[1][0], R[1][1], R[1][2], tCam[1]],
            [R[2][0], R[2][1], R[2][2], tCam[2]],
            [0, 0, 0, 1]
        ];
    }

    function getCamCenter(c2w) {
        return [c2w[0][3], c2w[1][3], c2w[2][3]];
    }

    function getWorldPoint(P_inv, u, v, depth) {
        const disparity = 1.0 / Math.max(depth, 1e-2);
        const pix = [u, v, 1, disparity];
        const pw = matvec4(P_inv, pix);
        const w = Math.max(Math.abs(pw[3]), 1e-6);
        return [pw[0]/w, pw[1]/w, pw[2]/w];
    }

    function getWorldDirection(P_inv, u, v) {
        const pix = [u, v, 1, 0];
        const pw = matvec4(P_inv, pix);
        const len = norm3(pw) + 1e-6;
        return [pw[0]/len, pw[1]/len, pw[2]/len];
    }

    function transformToQueryFrame3D(pointWorld, w2c) {
        const ph = [pointWorld[0], pointWorld[1], pointWorld[2], 1];
        const pc = matvec4(w2c, ph);
        const w = Math.max(Math.abs(pc[3]), 1e-4);
        return [pc[0]/w, pc[1]/w, pc[2]/w];
    }

    function transformToQueryFrameProj(pointWorld, P) {
        const ph = [pointWorld[0], pointWorld[1], pointWorld[2], 1];
        const pp = matvec4(P, ph);
        const z = Math.max(Math.abs(pp[2]), 1e-4);
        const w = Math.max(Math.abs(pp[3]), 1e-4);
        const len = norm3(pp) + 1e-6;
        const dir = [pp[0]/len, pp[1]/len, pp[2]/len];
        const disparity = w / z;
        return { dir, disparity };
    }

    function getFrequencies(numFreqs, maxPeriod, minPeriod) {
        const logMinFreq = Math.log(2 * Math.PI / maxPeriod);
        const logMaxFreq = Math.log(2 * Math.PI / minPeriod);
        const freqs = [];
        for (let i = 0; i < numFreqs; i++) {
            const t = numFreqs > 1 ? i / (numFreqs - 1) : 0;
            const logF = logMinFreq + t * (logMaxFreq - logMinFreq);
            freqs.push(Math.exp(logF));
        }
        return freqs;
    }

    function computeRopeCoeffs(positions, numFreqs, freqBase) {
        const allCos = [];
        const allSin = [];

        for (const [posName, posValues] of Object.entries(positions)) {
            let maxPeriod;
            let clampedPos = posValues;
            
            if (posName === 'p0' || posName === 'pd_3d') {
                maxPeriod = 1.0 * 4;
            } else if (posName === 'pd_dir' || posName === 'p0_dir') {
                maxPeriod = 2.0 * 4;
            } else if (posName === 'pd_disparity' || posName === 'p0_disparity') {
                maxPeriod = 20.0 * 4;
                clampedPos = posValues.map(v => Math.max(0, Math.min(20, v)));
            } else {
                maxPeriod = 1.0 * 4;
            }

            const minPeriod = maxPeriod / Math.pow(freqBase, numFreqs - 1);
            const freqs = getFrequencies(numFreqs, maxPeriod, minPeriod);

            for (const f of freqs) {
                for (const p of clampedPos) {
                    const angle = f * p;
                    allCos.push(Math.cos(angle));
                    allSin.push(Math.sin(angle));
                }
            }
        }

        return { cos: allCos, sin: allSin };
    }

    function applyRope(feats, cos, sin, inverse) {
        const halfDim = feats.length / 2;
        const x1 = feats.slice(0, halfDim);
        const x2 = feats.slice(halfDim);
        const out = new Array(feats.length);

        for (let i = 0; i < halfDim; i++) {
            if (inverse) {
                out[i] = x1[i] * cos[i] + x2[i] * sin[i];
                out[i + halfDim] = -x1[i] * sin[i] + x2[i] * cos[i];
            } else {
                out[i] = x1[i] * cos[i] - x2[i] * sin[i];
                out[i + halfDim] = x1[i] * sin[i] + x2[i] * cos[i];
            }
        }
        return out;
    }

    function computeAttentionScores(tx, tz, rotY, keyDepth, queryDepth) {
        const w2c_query = eye4();
        const w2c_key = createW2C(tx, tz, rotY);
        const c2w_query = invertSE3(w2c_query);
        const c2w_key = invertSE3(w2c_key);

        const K = [
            [IMAGE_WIDTH, 0, IMAGE_WIDTH/2],
            [0, IMAGE_HEIGHT, IMAGE_HEIGHT/2],
            [0, 0, 1]
        ];
        const Knorm = normalizeK(K, IMAGE_WIDTH, IMAGE_HEIGHT);
        const Kinv = invertK(Knorm);

        const P_query = matmul4x4(liftK(Knorm), w2c_query);
        const P_inv_query = matmul4x4(c2w_query, liftK(Kinv));
        const P_inv_key = matmul4x4(c2w_key, liftK(Kinv));

        const queryCamPos = getCamCenter(c2w_query);
        const keyCamPos = getCamCenter(c2w_key);

        const centerPatchX = Math.floor(PATCHES_X / 2);
        const centerPatchY = Math.floor(PATCHES_Y / 2);
        const centerX = (centerPatchX + 0.5) / PATCHES_X - 0.5;
        const centerY = (centerPatchY + 0.5) / PATCHES_Y - 0.5;

        const queryDepthPoint = getWorldPoint(P_inv_query, centerX, centerY, queryDepth);
        const queryRayDir = getWorldDirection(P_inv_query, centerX, centerY);
        const keyDepthPointWorld = getWorldPoint(P_inv_key, centerX, centerY, keyDepth);
        const keyRayDir = getWorldDirection(P_inv_key, centerX, centerY);

        const p0_query_in_query = transformToQueryFrame3D(queryCamPos, w2c_query);
        const pd_query_proj = transformToQueryFrameProj(queryDepthPoint, P_query);

        const positionsQ = {
            'p0': p0_query_in_query,
            'pd_dir': pd_query_proj.dir.slice(0, 2),
            'pd_disparity': [pd_query_proj.disparity]
        };
        const ropeQ = computeRopeCoeffs(positionsQ, NUM_ROPE_FREQS, FREQ_BASE);

        const scores = [];
        for (let py = 0; py < PATCHES_Y; py++) {
            for (let px = 0; px < PATCHES_X; px++) {
                const u = (px + 0.5) / PATCHES_X - 0.5;
                const v = (py + 0.5) / PATCHES_Y - 0.5;

                const keyPointWorld = getWorldPoint(P_inv_key, u, v, keyDepth);
                const p0_key_in_query = transformToQueryFrame3D(keyCamPos, w2c_query);
                const pd_key_proj = transformToQueryFrameProj(keyPointWorld, P_query);

                const positionsK = {
                    'p0': p0_key_in_query,
                    'pd_dir': pd_key_proj.dir.slice(0, 2),
                    'pd_disparity': [pd_key_proj.disparity]
                };
                const ropeK = computeRopeCoeffs(positionsK, NUM_ROPE_FREQS, FREQ_BASE);

                const q = new Array(HEAD_DIM).fill(1);
                const k = new Array(HEAD_DIM).fill(1);

                const qRope = applyRope(q, ropeQ.cos, ropeQ.sin, true);
                const kRope = applyRope(k, ropeK.cos, ropeK.sin, true);

                let dot = 0;
                for (let i = 0; i < HEAD_DIM; i++) {
                    dot += qRope[i] * kRope[i];
                }
                const score = dot / HEAD_DIM;
                scores.push(score);
            }
        }

        const scores2D = [];
        for (let py = 0; py < PATCHES_Y; py++) {
            const row = [];
            for (let px = 0; px < PATCHES_X; px++) {
                row.push(scores[py * PATCHES_X + px]);
            }
            scores2D.push(row);
        }

        return {
            scores2D,
            queryPos: queryCamPos,
            keyPos: keyCamPos,
            queryDepthPoint,
            keyDepthPoint: keyDepthPointWorld,
            queryRayDir,
            keyRayDir,
            w2c_key
        };
    }

    function createCameraFrustum(pos, c2w, color, scale = 0.5) {
        const R = [[c2w[0][0], c2w[0][1], c2w[0][2]],
                   [c2w[1][0], c2w[1][1], c2w[1][2]],
                   [c2w[2][0], c2w[2][1], c2w[2][2]]];

        const corners_cam = [
            [-0.3 * scale, -0.3 * scale, 0.5 * scale],
            [0.3 * scale, -0.3 * scale, 0.5 * scale],
            [0.3 * scale, 0.3 * scale, 0.5 * scale],
            [-0.3 * scale, 0.3 * scale, 0.5 * scale]
        ];

        const corners_world = corners_cam.map(c => {
            return [
                R[0][0]*c[0] + R[0][1]*c[1] + R[0][2]*c[2] + pos[0],
                R[1][0]*c[0] + R[1][1]*c[1] + R[1][2]*c[2] + pos[1],
                R[2][0]*c[0] + R[2][1]*c[1] + R[2][2]*c[2] + pos[2]
            ];
        });

        const x = [], y = [], z = [];

        for (const corner of corners_world) {
            x.push(pos[0], corner[0], null);
            y.push(pos[1], corner[1], null);
            z.push(pos[2], corner[2], null);
        }

        for (let i = 0; i < 4; i++) {
            const j = (i + 1) % 4;
            x.push(corners_world[i][0], corners_world[j][0], null);
            y.push(corners_world[i][1], corners_world[j][1], null);
            z.push(corners_world[i][2], corners_world[j][2], null);
        }

        return { x, y, z, color };
    }

    function updatePlots(tx, tz, rotY, keyDepth, queryDepth, plot3dId, plotHeatmapId) {
        const result = computeAttentionScores(tx, tz, rotY, keyDepth, queryDepth);

        // ========== 3D Plot ==========
        const c2w_query = eye4();
        const c2w_key = invertSE3(result.w2c_key);

        const queryFrustum = createCameraFrustum(result.queryPos, c2w_query, 'blue');
        const keyFrustum = createCameraFrustum(result.keyPos, c2w_key, 'red');

        const traces3D = [];

        traces3D.push({
            type: 'scatter3d',
            x: queryFrustum.x,
            y: queryFrustum.y,
            z: queryFrustum.z,
            mode: 'lines',
            line: { color: '#FF8C00', width: 4 },
            name: 'Query Camera'
        });

        traces3D.push({
            type: 'scatter3d',
            x: keyFrustum.x,
            y: keyFrustum.y,
            z: keyFrustum.z,
            mode: 'lines',
            line: { color: '#228B22', width: 4 },
            name: 'Key Camera'
        });

        const qp = result.queryPos;
        const qd = result.queryDepthPoint;
        const qDir = result.queryRayDir;
        const rayExtent = 10;
        const qRayEnd = [
            qp[0] + qDir[0] * rayExtent,
            qp[1] + qDir[1] * rayExtent,
            qp[2] + qDir[2] * rayExtent
        ];
        traces3D.push({
            type: 'scatter3d',
            x: [qp[0], qRayEnd[0]],
            y: [qp[1], qRayEnd[1]],
            z: [qp[2], qRayEnd[2]],
            mode: 'lines',
            line: { color: '#FF4500', width: 5 },
            name: 'Query Ray'
        });

        const kp = result.keyPos;
        const kd = result.keyDepthPoint;
        const kDir = result.keyRayDir;
        const kRayEnd = [
            kp[0] + kDir[0] * rayExtent,
            kp[1] + kDir[1] * rayExtent,
            kp[2] + kDir[2] * rayExtent
        ];
        traces3D.push({
            type: 'scatter3d',
            x: [kp[0], kRayEnd[0]],
            y: [kp[1], kRayEnd[1]],
            z: [kp[2], kRayEnd[2]],
            mode: 'lines',
            line: { color: '#006400', width: 5 },
            name: 'Key Ray'
        });

        traces3D.push({
            type: 'scatter3d',
            x: [qd[0]],
            y: [qd[1]],
            z: [qd[2]],
            mode: 'markers',
            marker: { size: 6, color: '#FFD700', symbol: 'circle', line: { width: 1, color: '#FF4500' } },
            name: 'Query Depth Point'
        });

        traces3D.push({
            type: 'scatter3d',
            x: [kd[0]],
            y: [kd[1]],
            z: [kd[2]],
            mode: 'markers',
            marker: { size: 6, color: '#90EE90', symbol: 'circle', line: { width: 1, color: '#006400' } },
            name: 'Key Depth Point'
        });

        const layout3D = {
            scene: {
                xaxis: { title: 'X', range: [-4, 4] },
                yaxis: { title: 'Y', range: [-2, 2] },
                zaxis: { title: 'Z (forward)', range: [-1, 6] },
                aspectmode: 'manual',
                aspectratio: { x: 1.3, y: 0.5, z: 1 },
                camera: {
                    eye: { x: 0, y: 0.5, z: -0.8 },
                    center: { x: 0, y: 0, z: 0 },
                    up: { x: 0, y: 1, z: 0 }
                }
            },
            margin: { l: 0, r: 0, t: 30, b: 0 },
            showlegend: true,
            legend: { x: 0, y: 1, bgcolor: 'rgba(255,255,255,0.8)' },
            uirevision: 'constant'
        };

        if (!plot3DInitialized) {
            Plotly.newPlot(plot3dId, traces3D, layout3D);
            plot3DInitialized = true;
        } else {
            Plotly.react(plot3dId, traces3D, layout3D);
        }

        // ========== Heatmap ==========
        const centerIdxX = Math.floor(PATCHES_X / 2);
        const centerIdxY = Math.floor(PATCHES_Y / 2);

        const traceHeatmap = [{
            type: 'heatmap',
            z: result.scores2D,
            colorscale: [
                [0, '#FFFFFF'],
                [0.25, '#C6DBEF'],
                [0.5, '#6BAED6'],
                [0.75, '#2171B5'],
                [1, '#08306B']
            ],
            colorbar: {
                title: { text: 'Attention Score', side: 'right' },
                thickness: 20,
                len: 0.9
            },
            x: Array.from({length: PATCHES_X}, (_, i) => i),
            y: Array.from({length: PATCHES_Y}, (_, i) => i),
            hoverongaps: false,
            hovertemplate: 'x: %{x}<br>y: %{y}<br>score: %{z:.4f}<extra></extra>'
        }];

        traceHeatmap.push({
            type: 'scatter',
            x: [centerIdxX],
            y: [centerIdxY],
            mode: 'markers',
            marker: { size: 18, color: 'white', symbol: 'x', line: { width: 3, color: 'red' } },
            name: 'Query Token',
            showlegend: false
        });

        const layoutHeatmap = {
            xaxis: {
                title: 'Patch X',
                dtick: 1,
                tickmode: 'linear',
                range: [-0.5, PATCHES_X - 0.5],
                scaleanchor: 'y',
                scaleratio: 1,
                constrain: 'domain'
            },
            yaxis: {
                title: 'Patch Y',
                dtick: 1,
                tickmode: 'linear',
                autorange: 'reversed',
                range: [PATCHES_Y - 0.5, -0.5],
                constrain: 'domain'
            },
            margin: { l: 60, r: 80, t: 30, b: 50 }
        };

        Plotly.react(plotHeatmapId, traceHeatmap, layoutHeatmap);
    }

    // Initialize the demo with given element IDs
    function initDemo(config) {
        const {
            txSliderId = 'demo-tx-slider',
            tzSliderId = 'demo-tz-slider',
            rotSliderId = 'demo-rot-slider',
            keyDepthSliderId = 'demo-key-depth-slider',
            queryDepthSliderId = 'demo-query-depth-slider',
            txValueId = 'demo-tx-value',
            tzValueId = 'demo-tz-value',
            rotValueId = 'demo-rot-value',
            keyDepthValueId = 'demo-key-depth-value',
            queryDepthValueId = 'demo-query-depth-value',
            plot3dId = 'demo-plot-3d',
            plotHeatmapId = 'demo-plot-heatmap'
        } = config || {};

        function onSliderChange() {
            const tx = parseFloat(document.getElementById(txSliderId).value);
            const tz = parseFloat(document.getElementById(tzSliderId).value);
            const rotY = parseFloat(document.getElementById(rotSliderId).value);
            const keyDepth = parseFloat(document.getElementById(keyDepthSliderId).value);
            const queryDepth = parseFloat(document.getElementById(queryDepthSliderId).value);

            document.getElementById(txValueId).textContent = tx.toFixed(1);
            document.getElementById(tzValueId).textContent = tz.toFixed(1);
            document.getElementById(rotValueId).textContent = rotY.toFixed(0) + '°';
            document.getElementById(keyDepthValueId).textContent = keyDepth.toFixed(1);
            document.getElementById(queryDepthValueId).textContent = queryDepth.toFixed(1);

            updatePlots(tx, tz, rotY, keyDepth, queryDepth, plot3dId, plotHeatmapId);
        }

        document.getElementById(txSliderId).addEventListener('input', onSliderChange);
        document.getElementById(tzSliderId).addEventListener('input', onSliderChange);
        document.getElementById(rotSliderId).addEventListener('input', onSliderChange);
        document.getElementById(keyDepthSliderId).addEventListener('input', onSliderChange);
        document.getElementById(queryDepthSliderId).addEventListener('input', onSliderChange);

        // Initial render
        updatePlots(1.0, 0.0, 0.0, 1.0, 1.0, plot3dId, plotHeatmapId);
    }

    // Export to global scope
    global.RayRoPEDemo = {
        init: initDemo,
        updatePlots: updatePlots
    };

})(window);
