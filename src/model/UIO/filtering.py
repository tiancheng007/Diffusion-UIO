from .ss_model import GSSModel, Uncertainty_Est_NN
from .diffusion_model import EstDiff
import torch
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
import time
import gc
import queue
import threading
from concurrent.futures import ThreadPoolExecutor


class Unknown_Input_Observer():
    def __init__(self, GSSModel: GSSModel):
        self.x_dim = 2
        self.y_dim = 1

        self.device = torch.device('cuda:0')
        self.device_index = 0
        self.seq_len = 15

        self.Observer_Model = GSSModel

        # For LPV UIO-based
        self.A1, self.A2, self.A3, self.Dd = self.Observer_Model.A_D_cal()

        self.gain_T = torch.tensor([[float(1), float(-1.2397)],
                                    [float(.0), float(.0)]]).to(self.device)

        self.gain_N = torch.tensor([[float(1.2397)],
                                    [float(1)]]).to(self.device)
        # 100%
        self.gain_M1 = torch.tensor([[float(1.8200), float(0.1172)], [float(0.0007), float(288.6321)]]).to(self.device)

        self.gain_M2 = torch.tensor([[float(1.0105), float(0.1191)], [float(-0.0004), float(288.6321)]]).to(self.device)

        self.gain_M3 = torch.tensor([[float(1.0105), float(0.1191)], [float(-0.0004), float(288.6321)]]).to(self.device)

        self.gain_L1 = torch.tensor([[float(-1.6809)],
                                     [float(-0.0007)]]).to(self.device)

        self.gain_L2 = torch.tensor([[float(-1.3221)], [float(-0.0004)]]).to(self.device)

        self.gain_L3 = torch.tensor([[float(-1.5748)],
                                     [float(-0.0013)]]).to(self.device)

        # 110%
        # self.gain_M1 = torch.tensor([[float(1.7477), float(0.1381)], [float(0.0005), float(398.7184)]]).to(self.device)
        #
        # self.gain_M2 = torch.tensor([[float(1.0107), float(0.1402)], [float(0.0001), float(398.7254)]]).to(self.device)
        #
        # self.gain_M3 = torch.tensor([[float(1.0107), float(0.1402)], [float(0.0001), float(398.7254)]]).to(self.device)
        #
        # self.gain_L1 = torch.tensor([[float(-1.7039)],
        #                              [float(0.0009)]]).to(self.device)
        #
        # self.gain_L2 = torch.tensor([[float(-1.3511)], [float(-0.0031)]]).to(self.device)
        #
        # self.gain_L3 = torch.tensor([[float(-1.6038)],
        #                              [float(-0.0085)]]).to(self.device)

        # 130%
        # self.gain_M1 = torch.tensor([[float(1.6247), float(0.1801)], [float(-0.0027), float(273.4502)]]).to(self.device)
        #
        # self.gain_M2 = torch.tensor([[float(1.0108), float(0.1825)], [float(0.0094), float(273.4234)]]).to(self.device)
        #
        # self.gain_M3 = torch.tensor([[float(1.0108), float(0.1825)], [float(0.0093), float(273.4233)]]).to(self.device)
        #
        # self.gain_L1 = torch.tensor([[float(-1.7334)],
        #                              [float(0.0029)]]).to(self.device)
        #
        # self.gain_L2 = torch.tensor([[float(-1.4053)], [float(-0.0127)]]).to(self.device)
        #
        # self.gain_L3 = torch.tensor([[float(-1.6580)],
        #                              [float(-0.0184)]]).to(self.device)

        # 85%
        # self.gain_M1 = torch.tensor([[float(1.9554), float(0.0866)], [float(0.0007), float(245.1151)]]).to(self.device)
        #
        # self.gain_M2 = torch.tensor([[float(1.0103), float(0.0877)], [float(-0.0007), float(245.1166)]]).to(self.device)
        #
        # self.gain_M3 = torch.tensor([[float(1.0103), float(0.0877)], [float(-0.0007), float(245.1166)]]).to(self.device)
        #
        # self.gain_L1 = torch.tensor([[float(-1.6378)],
        #                              [float(-0.0016)]]).to(self.device)
        #
        # self.gain_L2 = torch.tensor([[float(-1.2756)], [float(0.0008)]]).to(self.device)
        #
        # self.gain_L3 = torch.tensor([[float(-1.5281)],
        #                              [float(0.0004)]]).to(self.device)

        # 75%
        # self.gain_M1 = torch.tensor([[float(2.0689), float(0.0666)], [float(0.0008), float(244.0821)]]).to(self.device)
        #
        # self.gain_M2 = torch.tensor([[float(1.0101), float(0.0672)], [float(-0.0012), float(244.0842)]]).to(self.device)
        #
        # self.gain_M3 = torch.tensor([[float(1.0101), float(0.0672)], [float(-0.0012), float(244.0842)]]).to(self.device)
        #
        # self.gain_L1 = torch.tensor([[float(-1.5956)],
        #                              [float(-0.0020)]]).to(self.device)
        #
        # self.gain_L2 = torch.tensor([[float(-1.2415)], [float(0.0014)]]).to(self.device)
        #
        # self.gain_L3 = torch.tensor([[float(-1.4941)],
        #                              [float(0.0010)]]).to(self.device)

        # 115%
        # self.gain_M1 = torch.tensor([[float(1.7146), float(0.1489)], [float(-0.0003), float(562.6117)]]).to(self.device)
        #
        # self.gain_M2 = torch.tensor([[float(1.0107), float(0.1508)], [float(0.0021), float(562.6240)]]).to(self.device)
        #
        # self.gain_M3 = torch.tensor([[float(1.0107), float(0.1508)], [float(0.0020), float(562.6240)]]).to(self.device)
        #
        # self.gain_L1 = torch.tensor([[float(-1.7147)],
        #                              [float(0.0021)]]).to(self.device)
        #
        # self.gain_L2 = torch.tensor([[float(-1.3651)], [float(-0.0077)]]).to(self.device)
        #
        # self.gain_L3 = torch.tensor([[float(-1.6177)],
        #                              [float(-0.0190)]]).to(self.device)

        self.C_mat = torch.tensor([[float(.0), float(1)]]).to(self.device)

        # Pre-compute inverse for efficiency
        self.C_mat_Dd_pinv = torch.linalg.pinv(self.C_mat @ self.Dd)

        self.uncertain_net = Uncertainty_Est_NN()

        self.gen_net = EstDiff(model_dim=2, cond_dim=4, seq_len=15)

        self.init_state = torch.tensor([0., 0.]).reshape((-1, 1)).to(self.device)

        self.init_zeta = torch.zeros((self.x_dim, 1)).to(self.device)
        self.init_seq_cond = torch.zeros((1, self.seq_len, 4)).to(self.device)
        self.init_phi_k_hat = torch.zeros((self.x_dim, 1)).to(self.device)
        self.state_history = self.init_state.to(self.device).clone()

        self.sequence_position = 0
        self.max_sequence_length = 15

        self.current_buffer = None
        self.next_buffer = None
        self.computing_buffer = None

        self.gen_net_computing = False
        self.next_seq_cond = None
        self.compute_trigger_position = 7

        self.f_nn_zero = torch.zeros(2, 1).to(self.device)
        self.epsilon = torch.tensor(float(1e-12)).to(self.device)

        gc.disable()
        self.gc_counter = 0
        self.gc_interval = 100

        self.use_threading = True
        self.compute_stream = torch.cuda.Stream(device=self.device)

        if self.use_threading:
            self.compute_queue = queue.Queue(maxsize=2)
            self.result_queue = queue.Queue(maxsize=2)

            self.compute_thread = threading.Thread(target=self._compute_worker_thread, daemon=True)
            self.compute_thread.start()

            self.executor = ThreadPoolExecutor(max_workers=1)

        self._warmup_model()

        self.reset(clean_history=True)

    def _warmup_model(self):
        with torch.no_grad():
            dummy_input = torch.zeros((1, self.seq_len, 4)).to(self.device)
            normalized_input = F.normalize(dummy_input, p=2, dim=2, eps=1e-12)
            for _ in range(3):
                try:
                    _ = self.gen_net.predict(normalized_input, num_samples=1, steps=10)
                except:
                    pass

            torch.cuda.synchronize(self.device)

    def _compute_worker_thread(self):
        gc.disable()
        local_gc_counter = 0

        torch.cuda.set_device(self.device_index)

        while True:
            try:
                seq_cond = self.compute_queue.get(timeout=0.001)

                if seq_cond is None:
                    break

                local_gc_counter += 1
                if local_gc_counter % 50 == 0:
                    gc.collect(0)

                with torch.cuda.stream(self.compute_stream):
                    with torch.no_grad():
                        start_time = time.time()

                        normalized_input = F.normalize(seq_cond, p=2, dim=2, eps=1e-12)
                        result = self.gen_net.predict(normalized_input, num_samples=1, steps=10)

                        self.compute_stream.synchronize()

                        try:
                            self.result_queue.put_nowait(result)
                        except queue.Full:
                            try:
                                _ = self.result_queue.get_nowait()
                                self.result_queue.put_nowait(result)
                            except queue.Empty:
                                pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Compute worker error: {e}")
                import traceback
                traceback.print_exc()
                continue

    def reset(self, clean_history=False):
        self.zeta_post = self.init_zeta.detach().clone().to(self.device)
        self.state_post = self.init_state.detach().clone().to(self.device)
        self.phi_k_hat_post = self.init_phi_k_hat.detach().clone().to(self.device)

        self.zeta_post_m = self.init_zeta.detach().clone().to(self.device)
        self.state_post_m = self.init_state.detach().clone().to(self.device)
        self.phi_k_hat_post_m = self.init_phi_k_hat.detach().clone().to(self.device)

        self.seq_cond_post = self.init_seq_cond.detach().clone().to(self.device)
        self.iter_first = True
        self.state_history = torch.cat((self.state_history, self.state_post), axis=1).to(self.device)
        if clean_history:
            self.state_history = self.init_state.detach().clone().to(self.device)

        self.sequence_position = 0
        self.current_buffer = None
        self.next_buffer = None
        self.computing_buffer = None
        self.gen_net_computing = False
        self.next_seq_cond = None

        self.gc_counter = 0

        if self.use_threading:
            while not self.compute_queue.empty():
                try:
                    self.compute_queue.get_nowait()
                except:
                    break
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except:
                    break

    def _trigger_compute_if_needed(self, seq_cond_post, force=False):
        if self.gen_net_computing:
            return

        should_compute = force or (
                self.sequence_position >= self.compute_trigger_position and
                self.next_buffer is None
        )

        if should_compute:
            self.gen_net_computing = True
            self.next_seq_cond = seq_cond_post.clone()

            if self.use_threading:
                try:
                    self.compute_queue.put_nowait(self.next_seq_cond)
                except queue.Full:
                    try:
                        _ = self.compute_queue.get_nowait()
                        self.compute_queue.put_nowait(self.next_seq_cond)
                    except:
                        self.gen_net_computing = False

    def _check_compute_completion(self):
        if not self.gen_net_computing:
            return

        if self.use_threading:
            try:
                self.computing_buffer = self.result_queue.get_nowait()
                self.gen_net_computing = False

                if self.next_buffer is None:
                    self.next_buffer = self.computing_buffer
                    self.computing_buffer = None
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error checking compute completion: {e}")
                self.gen_net_computing = False

    def _get_current_f_nn_value(self):
        self._check_compute_completion()

        if self.sequence_position >= self.max_sequence_length:
            if self.next_buffer is not None:
                self.current_buffer = self.next_buffer
                self.next_buffer = self.computing_buffer
                self.computing_buffer = None
                self.sequence_position = 0
            else:
                self.sequence_position = 0

        if self.current_buffer is None:
            if self.next_buffer is not None:
                self.current_buffer = self.next_buffer
                self.next_buffer = None
                self.sequence_position = 0
            else:
                return self.f_nn_zero

        f_nn_dim_single = self.current_buffer[0, self.sequence_position, :].reshape(2, 1)

        self.sequence_position += 1

        return f_nn_dim_single

    def filtering_e2e_diff(self, observation, k):
        self.gc_counter += 1
        if self.gc_counter % self.gc_interval == 0:
            with torch.cuda.stream(torch.cuda.Stream(device=self.device)):
                gc.collect(0)

        with torch.no_grad():
            vx_obs = observation[0] + self.epsilon
            r_obs = observation[1] + self.epsilon

            self.seq_cond_post[0, :-1, :2] = self.seq_cond_post[0, 1:, :2]
            self.seq_cond_post[0, -1, 0] = observation[0]
            self.seq_cond_post[0, -1, 1] = observation[1]

            zeta_old = self.zeta_post
            zeta_old_m = self.zeta_post_m

            h1, h2, h3 = self.Observer_Model.h_cal(vx_obs)

            A_h = h1 * self.A1 + h2 * self.A2 + h3 * self.A3
            L_h = h1 * self.gain_L1 + h2 * self.gain_L2 + h3 * self.gain_L3
            M_h = h1 * self.gain_M1 + h2 * self.gain_M2 + h3 * self.gain_M3

            M_h_inv = torch.linalg.inv(M_h)

            y_k = r_obs.reshape(1, 1)

            x_hat_m = zeta_old_m + self.gain_N @ y_k
            phi_k_hat_m = A_h @ x_hat_m

            C_x_hat_m = self.C_mat @ x_hat_m
            ML_term = M_h_inv @ L_h
            y_minus_C_x_hat_m = y_k - C_x_hat_m

            zeta_new_m = self.gain_T @ phi_k_hat_m + ML_term @ y_minus_C_x_hat_m
            d_hat_m = self.C_mat_Dd_pinv @ (y_k - self.C_mat @ self.phi_k_hat_post_m)

            self.seq_cond_post[0, :-1, 2:] = self.seq_cond_post[0, 1:, 2:]
            self.seq_cond_post[0, -1, 2] = x_hat_m[0]
            self.seq_cond_post[0, -1, 3] = d_hat_m[0]

            if k >= self.seq_len:
                self._trigger_compute_if_needed(self.seq_cond_post)

                f_nn = self._get_current_f_nn_value()

                x_hat = zeta_old + self.gain_N @ y_k
                phi_k_hat = A_h @ x_hat + f_nn

                C_x_hat = self.C_mat @ x_hat
                y_minus_C_x_hat = y_k - C_x_hat
                zeta_new = self.gain_T @ phi_k_hat + ML_term @ y_minus_C_x_hat

                d_hat = self.C_mat_Dd_pinv @ (y_k - self.C_mat @ self.phi_k_hat_post)
            else:
                if k == self.seq_len - 1:
                    self._trigger_compute_if_needed(self.seq_cond_post, force=True)

                x_hat = x_hat_m
                phi_k_hat = phi_k_hat_m
                zeta_new = zeta_new_m
                d_hat = d_hat_m


            self.zeta_post = zeta_new
            self.state_post = x_hat
            self.phi_k_hat_post = phi_k_hat
            self.zeta_post_m = zeta_new_m
            self.state_post_m = x_hat_m
            self.phi_k_hat_post_m = phi_k_hat_m


            self.state_history = torch.cat((self.state_history, x_hat), dim=1)

            return x_hat[0], d_hat[0]

    def __del__(self):
        gc.enable()

        if self.use_threading and hasattr(self, 'compute_thread'):
            try:
                self.compute_queue.put(None)
                self.compute_thread.join(timeout=1.0)
            except:
                pass

            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)

    def filtering_e2e_label(self, observation):

        vx_obs = observation[0] + torch.tensor(float(1e-12)).to(self.device)
        r_obs = observation[1] + torch.tensor(float(1e-12)).to(self.device)

        zeta_old = self.zeta_post
        h1, h2, h3 = self.Observer_Model.h_cal(vx_obs)

        A_h = h1 * self.A1 + h2 * self.A2 + h3 * self.A3

        L_h = h1 * self.gain_L1 + h2 * self.gain_L2 + h3 * self.gain_L3

        M_h = h1 * self.gain_M1 + h2 * self.gain_M2 + h3 * self.gain_M3

        M_h_inv = torch.linalg.inv(M_h)

        y_k = r_obs.reshape(1, 1)

        x_hat = zeta_old + self.gain_N @ y_k

        f_nn_dim2 = self.uncertain_net(observation[:2].reshape(2, 1))

        phi_k_hat = A_h @ x_hat + f_nn_dim2

        zeta_new = self.gain_T @ phi_k_hat + M_h_inv @ L_h @ (y_k - self.C_mat @ x_hat)

        d_hat = torch.linalg.pinv(self.C_mat @ self.Dd) @ (y_k - self.C_mat @ self.phi_k_hat_post)

        assert not torch.any(torch.isnan(x_hat))

        self.zeta_post = zeta_new.clone()
        self.state_post = x_hat.clone()
        self.phi_k_hat_post = phi_k_hat.clone()
        self.state_history = torch.cat((self.state_history, x_hat.clone()), axis=1)

        return x_hat[0], d_hat[0], f_nn_dim2