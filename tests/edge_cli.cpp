#include "../src/cloud_circuit.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern char** environ;
namespace fs = std::filesystem;
using Json = nlohmann::json;

#ifdef EDGE_TEST_MLX
void write_mlx_fixture(const fs::path& directory, bool include_head = true);
#endif

namespace {
void require(bool condition, const std::string& detail) {
    if (!condition) throw std::runtime_error(detail);
}

struct Scratch {
    fs::path path;

    Scratch() {
        auto name = (fs::temp_directory_path() / "edge-test-XXXXXX").string();
        const auto* created = mkdtemp(name.data());
        require(created, "mkdtemp failed");
        path = created;
    }

    ~Scratch() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

class Socket {
    int fd_;
public:
    explicit Socket(int fd) : fd_(fd) { require(fd >= 0, "opening socket failed"); }
    ~Socket() { close(fd_); }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    int get() const { return fd_; }
};

struct Result {
    int code;
    std::string out;
    std::string error;
};

std::string read(const fs::path& path) {
    std::ifstream file(path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

class Cli {
    std::string binary_;
    Scratch files_;
public:
    explicit Cli(std::string binary) : binary_(std::move(binary)) {}

    pid_t start(std::vector<std::string> args, std::string_view input) const {
        std::ofstream(files_.path / "in") << input;
        args.insert(args.begin(), binary_);
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        posix_spawn_file_actions_t files;
        require(posix_spawn_file_actions_init(&files) == 0, "spawn file actions failed");
        const auto redirect = [&](int fd, const char* filename, int flags) {
            return posix_spawn_file_actions_addopen(&files, fd, (files_.path / filename).c_str(), flags, 0600);
        };
        const bool redirects_ok = redirect(STDIN_FILENO, "in", O_RDONLY) == 0
            && redirect(STDOUT_FILENO, "out", O_WRONLY | O_CREAT | O_TRUNC) == 0
            && redirect(STDERR_FILENO, "err", O_WRONLY | O_CREAT | O_TRUNC) == 0;
        pid_t pid = -1;
        const auto error = redirects_ok ? posix_spawn(&pid, binary_.c_str(), &files, nullptr, argv.data(), environ) : -1;
        posix_spawn_file_actions_destroy(&files);
        require(error == 0, "spawning edge CLI failed");
        return pid;
    }

    Result finish(pid_t pid) const {
        int status = 0;
        pid_t waited;
        do { waited = waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
        require(waited == pid && WIFEXITED(status), "edge CLI did not exit normally");
        return {WEXITSTATUS(status), read(files_.path / "out"), read(files_.path / "err")};
    }

    Result run(std::vector<std::string> args, std::string_view input = {}) const {
        return finish(start(std::move(args), input));
    }
};

void rejects(const Cli& cli, std::vector<std::string> args, std::string_view error) {
    const auto result = cli.run(std::move(args));
    require(result.code == 1 && result.out.empty() && result.error.find(error) != std::string::npos, result.error);
}

std::vector<Json> batch(const Result& result, std::size_t size) {
    require(result.code == 0 && result.error.empty(), "batch failed: " + result.error);
    std::vector<Json> rows;
    std::istringstream lines(result.out);
    for (std::string line; std::getline(lines, line);) {
        auto row = Json::parse(line);
        require(row.at("line") == rows.size() + 1, "wrong line number");
        rows.push_back(std::move(row));
    }
    require(rows.size() == size, "wrong number of results");
    return rows;
}

void test_arguments_and_batch(const Cli& cli, bool mlx) {
    const auto help = cli.run({"--help"});
    require(help.code == 0 && help.out.find("--batch") != std::string::npos, "missing usage");
    require(cli.run({}).out == help.out, "empty command should show usage");
    rejects(cli, {"--batch", "--prompt", "x"}, "either --batch or --prompt");
    rejects(cli, {"--prompt"}, "Missing value");
    rejects(cli, {"--prompt", ""}, "Supply --prompt");
    rejects(cli, {"--offline", "--prompt", "x"}, "--offline requires --model");
    rejects(cli, {"--other"}, "Unknown argument");
    rejects(cli, {"--max-tokens", "1x"}, "must be an integer");
    rejects(cli, {"--max-tokens", "0"}, "must be an integer");
    rejects(cli, {"--max-tokens", "1025"}, "must be an integer");
    rejects(cli, {"--batch", "--cloud-timeout-ms", "-1"}, "must be an integer");
    rejects(cli, {"--batch", "--cloud-cooldown-ms", "600001"}, "must be an integer");
    rejects(cli, {"--batch", "--cloud-cooldown-ms", "5"}, "requires cloud-first --batch with --model");
    rejects(cli, {"--demo", "--batch", "--model", "missing", "--cloud-cooldown-ms", "5"},
            "requires cloud-first --batch with --model");
    rejects(cli, {"--prompt", "x"}, "Set CLOUD_BASE_URL");
    if (!mlx) rejects(cli, {"--offline", "--model", "missing", "--prompt", "x"}, "EDGE_ENABLE_MLX");
    const auto demo = cli.run({"--demo"});
    require(demo.code == 0 && Json::parse(demo.out).at("source") == "fixture", "demo returned no fixture");
    const std::string input = "{\"prompt\":\"first\"}\n"
        "not json\n{\"prompt\":23}\n{}\n{\"prompt\":\"\"}\n"
        "{\"prompt\":\"second\",\"max_tokens\":0}\n"
        "{\"prompt\":\"second\",\"max_tokens\":1025}\n"
        "{\"prompt\":\"second\",\"max_tokens\":1.5}\n"
        + std::string(65537, 'x') + "\n{\"prompt\":\"last\",\"max_tokens\":1024}";
    const auto rows = batch(cli.run({"--demo", "--batch", "--max-tokens", "8"}, input), 10);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        require(rows[i].at("ok") == (i == 0 || i == 9), "wrong batch disposition at row " + std::to_string(i + 1));
    }
    require(rows[8].at("error") == "Request line exceeds 64 KiB.", "missing size-limit error");
    require(rows[9].at("text") == rows[0].at("text"), "batch did not recover after oversized record");
}

class Server {
    Socket socket_{socket(AF_INET, SOCK_STREAM, 0)};
    sockaddr_in address_{};
public:
    Server() {
        address_.sin_family = AF_INET;
        address_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        require(bind(socket_.get(), reinterpret_cast<sockaddr*>(&address_), sizeof(address_)) == 0, "bind failed");
        require(listen(socket_.get(), 4) == 0, "listen failed");
        socklen_t length = sizeof(address_);
        require(getsockname(socket_.get(), reinterpret_cast<sockaddr*>(&address_), &length) == 0, "getsockname failed");
        require(fcntl(socket_.get(), F_SETFD, FD_CLOEXEC) == 0, "marking listener close-on-exec failed");
    }

    std::string url() const { return "http://127.0.0.1:" + std::to_string(ntohs(address_.sin_port)) + "/v1//"; }

    bool has_pending_request() const {
        pollfd listener{socket_.get(), POLLIN, 0};
        const int result = poll(&listener, 1, 0);
        require(result >= 0, "testing for unexpected connections failed");
        return result > 0;
    }

    void exchange(std::string_view prompt, int tokens, std::string_view status, const std::string& content,
                  int wait_ms = 8000, std::size_t missing_bytes = 0) {
        pollfd listener{socket_.get(), POLLIN, 0};
        require(poll(&listener, 1, wait_ms) == 1, "edge CLI did not connect");
        Socket connection(accept(socket_.get(), nullptr, nullptr));
        const timeval timeout{5, 0};
        require(setsockopt(connection.get(), SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0,
                "receive timeout failed");
        std::string request;
        std::size_t body_start = std::string::npos;
        std::size_t content_len = 0;
        while (body_start == std::string::npos || request.size() < body_start + content_len) {
            char bytes[4096];
            const auto count = recv(connection.get(), bytes, sizeof(bytes), 0);
            require(count > 0, "truncated HTTP request");
            request.append(bytes, static_cast<std::size_t>(count));
            require(request.size() < 65536, "unexpectedly large HTTP request");
            if (body_start == std::string::npos && request.find("\r\n\r\n") != std::string::npos) {
                require(request.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"), "wrong HTTP request target");
                const auto end = request.find("\r\n\r\n");
                const auto header = request.find("\r\nContent-Length: ");
                require(header < end, "no Content-Length header");
                content_len = std::stoul(request.substr(header + 18));
                body_start = end + 4;
            }
        }
        require(request.find("\r\nAuthorization: Bearer test-key\r\n") < body_start, "auth header missing");
        const auto payload = Json::parse(request.substr(body_start));
        require(payload.at("model") == "test-model" && payload.at("stream") == false
                && payload.at("max_tokens") == tokens
                && payload.at("messages").at(0).at("content") == prompt, "unexpected request " + payload.dump());
        const auto response = "HTTP/1.1 " + std::string(status) + "\r\nContent-Type: application/json\r\n"
            "Connection: close\r\nContent-Length: " + std::to_string(content.size() + missing_bytes) + "\r\n\r\n" + content;
        std::size_t sent = 0;
        while (sent < response.size()) {
            const auto count = send(connection.get(), response.data() + sent, response.size() - sent, 0);
            if (count <= 0 && content.size() > 4 * 1024 * 1024) break; // Client stops reading at its cap.
            require(count > 0, "writing HTTP response failed");
            sent += static_cast<std::size_t>(count);
        }
    }
};

void point_at(const Server& server) {
    setenv("CLOUD_BASE_URL", server.url().c_str(), 1);
    setenv("CLOUD_MODEL", "test-model", 1);
    setenv("CLOUD_API_KEY", "test-key", 1);
    setenv("NO_PROXY", "127.0.0.1", 1);
    setenv("no_proxy", "127.0.0.1", 1);
}

void test_circuit_clock() {
    using namespace std::chrono_literals;
    const CloudCircuit::Clock::time_point start{};
    CloudCircuit circuit(100ms);
    require(!circuit.bypass(start), "fresh circuit bypassed");
    circuit.observe(429, start);
    require(!circuit.bypass(start), "rate limit opened the circuit");
    circuit.observe(503, start);
    require(circuit.bypass(start + 99ms), "circuit closed before the deadline");
    require(!circuit.bypass(start + 100ms), "probe blocked at the deadline");
    circuit.observe(503, start + 100ms);
    require(circuit.bypass(start + 199ms), "failed probe did not reopen the circuit");
    circuit.observe(200, start + 200ms);
    require(!circuit.bypass(start + 200ms), "successful probe did not clear the circuit");
    CloudCircuit disabled(0ms);
    disabled.observe(503, start);
    require(!disabled.bypass(start), "default settings opened a circuit");
}

void test_cloud(const Cli& cli) {
    test_circuit_clock();
    Server server;
    point_at(server);
    const auto pid = cli.start({"--batch", "--max-tokens", "16"},
        "{\"prompt\":\"one\"}\n{\"prompt\":\"two\",\"max_tokens\":8}\n"
        "{\"prompt\":\"three\"}\n{\"prompt\":\"four\"}\n{\"prompt\":\"five\"}\n"
        "{\"prompt\":\"truncated unavailable\"}\n{\"prompt\":\"truncated success\"}\n");
    server.exchange("one", 16, "200 OK", R"({"choices":[{"message":{"content":"upstream"}}]})");
    server.exchange("two", 8, "503 Service Unavailable", "{}");
    server.exchange("three", 16, "429 Too Many Requests", "{}");
    server.exchange("four", 16, "200 OK", R"({"choices":[]})");
    server.exchange("five", 16, "200 OK", std::string(4 * 1024 * 1024 + 1, 'a'));
    server.exchange("truncated unavailable", 16, "503 Service Unavailable", "{}", 8000, 1);
    server.exchange("truncated success", 16, "200 OK", R"({"choices":[{"message":{"content":"partial"}}]})", 8000, 1);
    const auto rows = batch(cli.finish(pid), 7);
    require(rows[0].at("source") == "cloud" && rows[0].at("text") == "upstream", rows[0].dump());
    require(rows[1].at("error").get<std::string>().find("Pass --model") != std::string::npos, rows[1].dump());
    require(rows[2].at("error") == "Cloud HTTP 429", rows[2].dump());
    require(rows[3].at("ok") == false, rows[3].dump());
    require(rows[4].at("error").get<std::string>().starts_with("Cloud request failed:"), rows[4].dump());
    require(rows[5].at("error").get<std::string>().find("Pass --model") != std::string::npos, rows[5].dump());
    require(rows[6].at("error").get<std::string>().starts_with("Cloud request failed:"), rows[6].dump());
}

#ifdef EDGE_TEST_MLX
void test_dense_checkpoint(const Cli& cli) {
    Scratch scratch;
    const auto model = scratch.path / "dense";
    const std::vector<std::string> args{"--offline", "--batch", "--model", model.string(), "--max-tokens", "3"};
    const auto request = R"({"prompt":"hello "})";
    const auto config_path = model / "config.json";
    const auto weights = model / "model.safetensors";
    write_mlx_fixture(model);
    const auto rows = batch(cli.run(args, R"({"prompt":"hello","max_tokens":20}
{"prompt":"hello"}
{"prompt":"hello","max_tokens":1})"), 3);
    require(rows[0].at("error") == "Prompt and generation budget exceed the model context.", rows[0].dump());
    require(rows[1].at("text") == "yes yes yes" && rows[1].at("local_metrics").at("model_reused") == true,
            rows[1].dump());
    require(rows[2].at("local_metrics").at("generated_tokens") == 1, rows[2].dump());

    const auto expect_error = [&](std::string_view error) {
        const auto row = batch(cli.run(args, request), 1).front();
        require(row.at("ok") == false && row.at("error").get<std::string>().find(error) != std::string::npos, row.dump());
    };
    auto config = Json::parse(read(config_path));
    config["eos_token_id"] = 2;
    std::ofstream(config_path) << config.dump();
    const auto stopped = batch(cli.run(args, request), 1).front();
    require(stopped.at("text") == "" && stopped.at("local_metrics").at("generated_tokens") == 0, stopped.dump());

    config["model_type"] = "unknown";
    std::ofstream(config_path) << config.dump();
    expect_error("supported Qwen2 or Llama");
    config["model_type"] = "qwen2";
    config["quantization"] = {{"mode", "affine"}, {"bits", 8}, {"group_size", 64}};
    std::ofstream(config_path) << config.dump();
    expect_error("supports dense weights and MLX affine 4-bit");
    config.erase("quantization");
    config["rope_scaling"] = {{"rope_type", "unsupported"}};
    std::ofstream(config_path) << config.dump();
    expect_error("Unsupported RoPE scaling");
    config["rope_scaling"] = {{"rope_type", "llama3"}, {"factor", 0}, {"low_freq_factor", 1},
                              {"high_freq_factor", 4}, {"original_max_position_embeddings", 16}};
    std::ofstream(config_path) << config.dump();
    expect_error("Invalid Llama RoPE scaling");
    config.erase("rope_scaling");
    config["vocab_size"] = 1;
    std::ofstream(config_path) << config.dump();
    expect_error("Tokenizer and model vocabulary differ");
    config["vocab_size"] = 4;
    std::ofstream(config_path) << config.dump();

    const auto other_shard = model / "model-duplicate.safetensors";
    fs::copy_file(weights, other_shard);
    expect_error("Duplicate tensors across safetensors shards");
    fs::remove(other_shard);
    fs::rename(weights, model / "ignored.safetensors");
    expect_error("No model*.safetensors weights");
    fs::remove(config_path);
    expect_error("Cannot open");
    fs::remove_all(model);
    write_mlx_fixture(model, false);
    expect_error("Missing model tensor lm_head.weight");
}
#endif

void test_mlx(const Cli& cli, const std::string& model, bool cooldown) {
    const std::string first = "Reply with just the number. What is 2 plus 2?";
    const std::string second = "Reply with just the number. What is 3 plus 3?";
    Server server;
    point_at(server);
    const auto request = [](const std::string& text) { return Json{{"prompt", text}}.dump() + "\n"; };
    std::vector<std::string> args{"--batch", "--model", model, "--max-tokens", "8"};
    if (cooldown) {
        args.insert(args.end(), {"--cloud-cooldown-ms", "600000", "--cloud-timeout-ms", "2000"});
        const auto probe = cli.run(args, "");
        require(probe.code == 0, "cooldown flag rejected: " + probe.error);
    }
    const auto pid = cli.start(std::move(args), request(first) + "malformed\n" + request(second));
    server.exchange(first, 8, "503 Service Unavailable", "{}", 8000, cooldown ? 1 : 0);
    if (!cooldown) server.exchange(second, 8, "503 Service Unavailable", "{}", 60000);
    const auto rows = batch(cli.finish(pid), 3);
    for (const std::size_t i : {0, 2}) {
        const auto& row = rows[i];
        const auto& metrics = row.at("local_metrics");
        const bool skipped = cooldown && i == 2;
        require(row.at("source") == "mlx" && row.at("ok") == true, row.dump());
        require(skipped ? row.at("cloud_status").is_null() : row.at("cloud_status") == 503, row.dump());
        require(row.at("fallback_reason") == (skipped ? "circuit_open" : "cloud_503"), row.dump());
        require(metrics.at("model_reused") == (i == 2) && metrics.at("generated_tokens").get<int>() > 0,
                row.dump());
        require(i == 0 ? metrics.at("load_ms").get<double>() > 0.0 : metrics.at("load_ms") == 0.0, row.dump());
    }
    require(rows[0].at("text").get<std::string>().find('4') != std::string::npos, rows[0].dump());
    require(rows[1].at("ok") == false, rows[1].dump());
    require(rows[2].at("text").get<std::string>().find('6') != std::string::npos, rows[2].dump());
    require(!server.has_pending_request(), "the skipped request reached the cloud server");
    const auto offline = batch(cli.run({"--batch", "--offline", "--model", model, "--max-tokens", "1"},
                                   request(first) + request("<|reserved|>")), 2);
    require(offline[0].at("source") == "mlx" && offline[0].at("cloud_status").is_null()
            && offline[0].at("local_metrics").at("generated_tokens") == 1, offline[0].dump());
    require(offline[1].at("error") == "Prompt contains a reserved chat token prefix.", offline[1].dump());
}
} // namespace

int main(int argc, char** argv) try {
    require(argc >= 3, "expected the edge executable and test mode");
    signal(SIGPIPE, SIG_IGN);
    unsetenv("CLOUD_BASE_URL");
    unsetenv("CLOUD_MODEL");
    unsetenv("CLOUD_API_KEY");
    const Cli cli(argv[1]);
    const std::string_view mode = argv[2];
    if (mode == "mlx" || mode == "mlx-cooldown") {
        require(argc == 4, "missing test model");
        test_mlx(cli, argv[3], mode == "mlx-cooldown");
#ifdef EDGE_TEST_MLX
    } else if (mode == "dense") {
        test_dense_checkpoint(cli);
#endif
    } else if (mode == "cloud") {
        test_cloud(cli);
    } else {
        test_arguments_and_batch(cli, mode == "mlx-arguments");
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
