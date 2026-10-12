package mage.player.ai.policy;

import com.google.gson.*;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** JSON v2 client. A failed policy call fails the game; there is no search fallback. */
public final class PolicyClient {
    private final String endpoint;
    private final Gson gson = new GsonBuilder().serializeNulls().create();
    public final Map<String, Integer> cardIds = new HashMap<>();
    public final String catalogDigest;

    public PolicyClient(String endpoint) {
        this.endpoint = endpoint;
        JsonObject health = request("GET", "/healthz", null);
        if (health.get("schema_version").getAsInt() != 2) {
            throw new IllegalStateException("XMage policy player requires a v2 inference service");
        }
        JsonObject catalog = request("GET", "/catalog", null);
        catalogDigest = catalog.get("catalog_digest").getAsString();
        for (JsonElement entry : catalog.getAsJsonObject("catalog").getAsJsonArray("cards")) {
            JsonObject card = entry.getAsJsonObject();
            String name = card.get("name").getAsString();
            if (cardIds.put(name, card.get("id").getAsInt()) != null) {
                throw new IllegalStateException("XMage name mapping requires unique names in the card catalog");
            }
        }
    }

    public JsonObject request(String method, String path, Object body) {
        HttpURLConnection connection = null;
        try {
            connection = (HttpURLConnection) new URL(endpoint + path).openConnection();
            connection.setRequestMethod(method);
            connection.setConnectTimeout(5000);
            connection.setReadTimeout(30000);
            if (body != null) {
                connection.setDoOutput(true);
                connection.setRequestProperty("Content-Type", "application/json");
                byte[] bytes = gson.toJson(body).getBytes(StandardCharsets.UTF_8);
                connection.setFixedLengthStreamingMode(bytes.length);
                try (OutputStream output = connection.getOutputStream()) { output.write(bytes); }
            }
            int status = connection.getResponseCode();
            InputStream input = status < 400 ? connection.getInputStream() : connection.getErrorStream();
            if (input == null) throw new IOException("empty HTTP response: " + status);
            JsonObject response;
            try (Reader reader = new InputStreamReader(input, StandardCharsets.UTF_8)) {
                response = gson.fromJson(reader, JsonObject.class);
            }
            if (status != 200) throw new IOException("HTTP " + status + ": " + response);
            return response;
        } catch (IOException error) {
            throw new IllegalStateException("policy endpoint failed: " + endpoint + path, error);
        } finally {
            if (connection != null) connection.disconnect();
        }
    }
}
